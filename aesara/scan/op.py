"""This module provides the `Scan` `Op`.

Memory reuse in scan
--------------------

To reduce the number of memory allocations and copies associated with calling
the inner function and recovering the outputs at every iteration, Scan uses a
memory pre-allocation mechanism for some of its outputs. Instead of repeatedly
calling the inner function and copying the outputs to designated locations,
it tries to make the inner function write the outputs directly to the
designated locations.

This is achieved by initializing, at every iteration, the output storage
of the inner function with references to previously allocated memory. Other
than the code in the Python and Cython backends to do this and to ensure that
the pre-allocated memory has been used, the memory pre-allocation mechanism
relies on the following elements to work properly :
- In make_thunk(), when compiling the inner function, the borrow flag must
  be set to False for the inputs. This will prevent aliasing between the
  inputs and the outputs of the inner function which could lead to invalid
  results.
- In make_thunk(), again, the borrow flag must be set to True for the outputs.
  This will make Aesara consider the output storages as persistent and make
  Aesara provide them as pre-allocated storage to the ops that compute the
  outputs of the inner function instead of letting these ops allocate their
  own output storage.
- The ops that produce the outputs of the inner function must be prevented
  from working inplace because if they do, they're not using the pre-allocated
  storage. This is achieved by including the optimization
  'add_no_output_from_inplace' to the compilation mode used by scan. It
  prevents other optimizations from altering the graph such that outputs are
  produced by inplace operations.
- The ScanSaveMem optimization, whose goal is to limit the amount of memory
  used by scan, needs to allocate buffers large enough to be able, at every
  iteration, to simultaneously read the needed previous states and storing
  the new states. Before the memory reuse feature, the buffers could be
  smaller because, often, Scan only needed buffers large enough to read the
  needed previous states. This is because all the outputs of the inner
  function were computed before any of them was stored in the buffers. Now,
  the outputs are stored as they are computed which means that, if the buffer
  is too small, computing an output can overwrite an input that is still
  needed to compute another output.

"""

import dataclasses
import logging
import time
from collections import OrderedDict
from copy import copy
from itertools import chain, product
from typing import Callable, List, Optional, Union

import numpy as np

import aesara
from aesara import tensor as at
from aesara.compile.builders import construct_nominal_fgraph, infer_shape
from aesara.compile.function.pfunc import pfunc
from aesara.compile.io import In, Out
from aesara.compile.mode import Mode, get_default_mode, get_mode
from aesara.compile.profiling import register_profiler_printer
from aesara.configdefaults import config
from aesara.gradient import DisconnectedType, NullType, Rop, grad, grad_undefined
from aesara.graph.basic import (
    Apply,
    Variable,
    clone_replace,
    equal_computations,
    graph_inputs,
    io_connection_pattern,
)
from aesara.graph.features import NoOutputFromInplace
from aesara.graph.op import HasInnerGraph, Op
from aesara.graph.utils import InconsistencyError, MissingInputError
from aesara.link.c.basic import CLinker
from aesara.link.c.exceptions import MissingGXX
from aesara.link.utils import raise_with_op
from aesara.printing import op_debug_information
from aesara.scan.utils import ScanProfileStats, Validator, forced_replace, safe_new
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.math import minimum
from aesara.tensor.shape import Shape_i
from aesara.tensor.type import TensorType, integer_dtypes
from aesara.tensor.var import TensorVariable


# Logging function for sending warning or info
_logger = logging.getLogger("aesara.scan.op")


err_msg1 = (
    "When compiling the inner function of scan (the "
    "function called by scan in each of its iterations) "
    "the following error has been encountered: The "
    "%s %s (argument number %d) has dtype "
    "%s and %d dimension(s). The corresponding variable "
    "in the inner function of scan %s "
    "however has dtype %s and %d dimension(s). This "
    "variable in the inner function of scan should "
    "have the same dtype and one fewer dimension "
    "compared to its corresponding variable in the initial "
    "state (outputs_info in scan nomenclature). For example, "
    "if the inner function of scan returns a vector "
    "of size d and scan uses the values of "
    "the previous time-step, then the initial state in scan "
    "should be a matrix of shape (1, d). "
    "The first dimension of this "
    "matrix corresponds to the number of previous time-steps "
    "that scan uses in each of its iterations. "
    "In order to solve this issue if the two variable currently "
    "have the same dimensionality, you can increase the "
    "dimensionality of the varialbe in the initial state of scan "
    "by using dimshuffle or shape_padleft. "
)
err_msg2 = (
    "When compiling the inner function of scan the "
    "following error has been encountered: The "
    "initial state (`outputs_info` in scan nomenclature) "
    "of variable %s (argument number %d) "
    "has dtype %s, while the result of the inner function "
    "(`fn`) has dtype %s. This can happen if the inner "
    "function of scan results in an upcast or downcast."
)
err_msg3 = (
    "When compiling the inner function of scan (the "
    "function called by scan in each of its iterations) "
    "the following error has been encountered: The "
    "initial state (`outputs_info` in scan nomenclature) "
    "of variable %s (argument number %d) has %d dimension(s), "
    "while the corresponding variable in the result of the inner "
    "function of scan (`fn`) has %d dimension(s) (it should "
    "be one less than the initial state). For example, "
    "if the inner function of scan returns a vector "
    "of size d and scan uses the values of "
    "the previous time-step, then the initial state in scan "
    "should be a matrix of shape (1, d). "
    "The first dimension of this "
    "matrix corresponds to the number of previous time-steps "
    "that scan uses in each of its iterations. "
    "In order to solve this issue if the two varialbe currently "
    "have the same dimensionality, you can increase the "
    "dimensionality of the variable in the initial state of scan "
    "by using dimshuffle or shape_padleft. "
)


def check_broadcast(v1, v2):
    """Checks that the broadcast pattern of v1 and v2.

    Controls that the broadcast pattern of the variable provided as
    input to `scan` matches the broadcast pattern provided in
    `output_info`. It raises an error when they don't match. The
    typical case is when the user provides either the input or the
    `output_info` (but not both) with a dimension fixed to 1,
    which may wrongly be interpreted as broadcastable.

    """
    if not isinstance(v1.type, TensorType) and not isinstance(v2.type, TensorType):
        return

    msg = (
        "The broadcast pattern of the output of scan (%s) is "
        "inconsistent with the one provided in `output_info` "
        "(%s). The output on axis %d is `%r`, but it is `%r` on "
        "axis %d in `output_info`. This can happen if one of the "
        "dimension is fixed to 1 in the input, while it is still "
        "variable in the output, or vice-verca. You have to make "
        "them consistent, e.g. using aesara.tensor."
        "{unbroadcast, specify_broadcastable}."
    )
    size = min(v1.type.ndim, v2.type.ndim)
    for n, (b1, b2) in enumerate(
        zip(v1.type.broadcastable[-size:], v2.type.broadcastable[-size:])
    ):
        if b1 != b2:
            a1 = n + size - v1.type.ndim + 1
            a2 = n + size - v2.type.ndim + 1
            raise TypeError(msg % (v1.type, v2.type, a1, b1, b2, a2))


def copy_var_format(var, as_var):
    """
    This functions ensures that ``var`` has the same dtype as ``as_var`` as
    well as calling `filter_variable` to make sure they are both `TensorType`.

    It internally deals with the corner case where ``inp.ndim + 1 = out.ndim``.

    """
    if not hasattr(var, "dtype"):
        return var
    rval = var
    if rval.type.dtype != as_var.type.dtype:
        rval = rval.astype(as_var.type.dtype)
    if rval.ndim == as_var.ndim:
        rval = as_var.type.filter_variable(rval)
    else:
        tmp = as_var.type.clone(
            shape=(tuple(var.type.shape[:1]) + tuple(as_var.type.shape))
        )
        rval = tmp.filter_variable(rval)
    return rval


@dataclasses.dataclass(frozen=True)
class ScanInfo:
    n_seqs: int
    mit_mot_in_slices: tuple
    mit_mot_out_slices: tuple
    mit_sot_in_slices: tuple
    sit_sot_in_slices: tuple
    n_nit_sot: int
    n_shared_outs: int
    n_non_seqs: int
    as_while: bool

    @property
    def n_mit_mot(self):
        return len(self.mit_mot_in_slices)

    @property
    def n_mit_mot_outs(self):
        return sum(len(x) for x in self.mit_mot_out_slices)

    @property
    def n_mit_sot(self):
        return len(self.mit_sot_in_slices)

    @property
    def n_sit_sot(self):
        return len(self.sit_sot_in_slices)

    @property
    def tap_array(self):
        return self.mit_mot_in_slices + self.mit_sot_in_slices + self.sit_sot_in_slices

    @property
    def n_inner_inputs(self):
        return (
            self.n_seqs
            + sum(len(x) for x in self.mit_mot_in_slices)
            + sum(len(x) for x in self.mit_sot_in_slices)
            + self.n_sit_sot
            + self.n_shared_outs
            + self.n_non_seqs
        )

    @property
    def n_inner_outputs(self):
        return (
            self.n_mit_mot_outs
            + self.n_mit_sot
            + self.n_sit_sot
            + self.n_nit_sot
            + self.n_shared_outs
            + int(self.as_while)
        )

    @property
    def n_outer_inputs(self):
        return (
            1
            + self.n_seqs
            + self.n_mit_mot
            + self.n_mit_sot
            + self.n_sit_sot
            + self.n_nit_sot
            + self.n_shared_outs
            + self.n_non_seqs
        )

    @property
    def n_outer_outputs(self):
        return (
            len(self.mit_mot_out_slices)
            + self.n_mit_sot
            + self.n_sit_sot
            + self.n_nit_sot
            + self.n_shared_outs
        )


TensorConstructorType = Callable[[List[bool], Union[str, np.generic]], TensorType]


class ScanMethodsMixin:
    def inner_seqs(self, list_inputs):
        # Given the list of inner inputs this function grabs those
        # corresponding to sequences
        return list_inputs[: self.info.n_seqs]

    def outer_seqs(self, list_inputs):
        # Given the list of outer inputs this function grabs those
        # corresponding to sequences
        return list_inputs[1 : 1 + self.info.n_seqs]

    def inner_mitmot(self, list_inputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_in_slices)
        return list_inputs[self.info.n_seqs : self.info.n_seqs + n_taps]

    def outer_mitmot(self, list_inputs):
        return list_inputs[
            1 + self.info.n_seqs : 1 + self.info.n_seqs + self.info.n_mit_mot
        ]

    def inner_mitmot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_out_slices)
        return list_outputs[:n_taps]

    def outer_mitmot_outs(self, list_outputs):
        return list_outputs[: self.info.n_mit_mot]

    def inner_mitsot(self, list_inputs):
        n_mitmot_taps = sum(len(x) for x in self.info.mit_mot_in_slices)
        ntaps_upto_sit_sot = n_mitmot_taps + sum(
            len(x) for x in self.info.mit_sot_in_slices
        )
        return list_inputs[
            self.info.n_seqs + n_mitmot_taps : self.info.n_seqs + ntaps_upto_sit_sot
        ]

    def outer_mitsot(self, list_inputs):
        offset = 1 + self.info.n_seqs + self.info.n_mit_mot
        return list_inputs[offset : offset + self.info.n_mit_sot]

    def inner_mitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_out_slices)
        return list_outputs[n_taps : n_taps + self.info.n_mit_sot]

    def outer_mitsot_outs(self, list_outputs):
        return list_outputs[
            self.info.n_mit_mot : self.info.n_mit_mot + self.info.n_mit_sot
        ]

    def inner_sitsot(self, list_inputs):
        n_taps_upto_sit_sot = sum(
            len(x)
            for x in chain(self.info.mit_mot_in_slices, self.info.mit_sot_in_slices)
        )
        offset = self.info.n_seqs + n_taps_upto_sit_sot
        return list_inputs[offset : offset + self.info.n_sit_sot]

    def outer_sitsot(self, list_inputs):
        offset = 1 + self.info.n_seqs + self.info.n_mit_mot + self.info.n_mit_sot
        return list_inputs[offset : offset + self.info.n_sit_sot]

    def inner_sitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_out_slices)
        offset = self.info.n_mit_sot + n_taps
        return list_outputs[offset : offset + self.info.n_sit_sot]

    def outer_sitsot_outs(self, list_outputs):
        offset = self.info.n_mit_mot + self.info.n_mit_sot
        return list_outputs[offset : offset + self.info.n_sit_sot]

    def outer_nitsot(self, list_inputs):
        offset = (
            1
            + self.info.n_seqs
            + self.info.n_mit_mot
            + self.info.n_mit_sot
            + self.info.n_sit_sot
            + self.info.n_shared_outs
        )
        return list_inputs[offset : offset + self.info.n_nit_sot]

    def inner_nitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_out_slices)
        offset = self.info.n_mit_sot + n_taps + self.info.n_sit_sot
        return list_outputs[offset : offset + self.info.n_nit_sot]

    def outer_nitsot_outs(self, list_outputs):
        offset = self.info.n_mit_mot + self.info.n_mit_sot + self.info.n_sit_sot
        return list_outputs[offset : offset + self.info.n_nit_sot]

    def inner_shared(self, list_inputs):
        n_taps_upto_sit_sot = sum(
            len(x)
            for x in chain(self.info.mit_mot_in_slices, self.info.mit_sot_in_slices)
        )
        offset = self.info.n_seqs + n_taps_upto_sit_sot + self.info.n_sit_sot
        return list_inputs[offset : offset + self.info.n_shared_outs]

    def outer_shared(self, list_inputs):
        offset = (
            1
            + self.info.n_seqs
            + self.info.n_mit_mot
            + self.info.n_mit_sot
            + self.info.n_sit_sot
        )
        return list_inputs[offset : offset + self.info.n_shared_outs]

    def inner_shared_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.info.mit_mot_out_slices)
        offset = (
            self.info.n_mit_sot + n_taps + self.info.n_sit_sot + self.info.n_nit_sot
        )
        return list_outputs[offset : offset + self.info.n_shared_outs]

    def outer_shared_outs(self, list_outputs):
        offset = (
            self.info.n_mit_mot
            + self.info.n_mit_sot
            + self.info.n_sit_sot
            + self.info.n_nit_sot
        )
        return list_outputs[offset : offset + self.info.n_shared_outs]

    def inner_non_seqs(self, list_inputs):
        n_taps_upto_sit_sot = sum(
            len(x)
            for x in chain(self.info.mit_mot_in_slices, self.info.mit_sot_in_slices)
        )
        offset = (
            self.info.n_seqs
            + n_taps_upto_sit_sot
            + self.info.n_sit_sot
            + self.info.n_shared_outs
        )
        return list_inputs[offset:]

    def outer_non_seqs(self, list_inputs):
        offset = (
            1
            + self.info.n_seqs
            + self.info.n_mit_mot
            + self.info.n_mit_sot
            + self.info.n_sit_sot
            + self.info.n_nit_sot
            + self.info.n_shared_outs
        )
        return list_inputs[offset:]

    def get_oinp_iinp_iout_oout_mappings(self):
        """
        Compute and return dictionary mappings between the inputs and
        outputs of the inner function and the inputs and outputs of the Scan
        node in the outer graph.

        The return value is a dictionary in which the keys are the names of
        the individual mappings and the values are the mapping dictionaries
        themselves. In dictionaries representing mappings to outer variables,
        the values are individual integer indices. In dictionaries
        representing mappings to inner variables, the values are sequences of
        indices because multiple inner variables can be associated with the
        same state.

        """
        # Lists for outer variables contain individual indices, lists for
        # inner variables contain sequences of indices because many inner
        # variables can be associated with the same outer variable. The list
        # and indices are initialized already containing the data associated
        # with the timestep index, the first outer input.
        outer_input_indices = [0]
        inner_input_indices = [[]]
        inner_output_indices = [[]]
        outer_output_indices = [-1]

        outer_iidx = 1
        inner_iidx = 0
        inner_oidx = 0
        outer_oidx = 0

        # Handle sequences inputs
        for i in range(self.info.n_seqs):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([])
            outer_output_indices.append(-1)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 0
            outer_oidx += 0

        # Handle mitmots, mitsots and sitsots variables
        for i, tap in enumerate(
            chain(
                self.info.mit_mot_in_slices,
                self.info.mit_sot_in_slices,
                self.info.sit_sot_in_slices,
            )
        ):
            nb_input_taps = len(tap)

            if i < self.info.n_mit_mot:
                nb_output_taps = len(self.info.mit_mot_out_slices[i])
            else:
                nb_output_taps = 1

            outer_input_indices.append(outer_iidx)
            inner_input_indices.append(
                list(range(inner_iidx, inner_iidx + nb_input_taps))
            )
            inner_output_indices.append(
                list(range(inner_oidx, inner_oidx + nb_output_taps))
            )
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += nb_input_taps
            inner_oidx += nb_output_taps
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx += self.info.n_shared_outs

        # Handle nitsots variables
        for i in range(self.info.n_nit_sot):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([])
            inner_output_indices.append([inner_oidx])
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += 0
            inner_oidx += 1
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx -= self.info.n_shared_outs + self.info.n_nit_sot

        # Handle shared states
        for i in range(self.info.n_shared_outs):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([inner_oidx])
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 1
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx += self.info.n_nit_sot

        # Handle non-sequence inputs
        # Note : the number of non-sequence inputs is not stored in self.info
        # so it has to be inferred from the number of inner inputs that remain
        # to be handled
        for i in range(len(self.inner_inputs) - inner_iidx):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([])
            outer_output_indices.append(-1)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 0
            outer_oidx += 0

        # With the global mapping inferred, the individual mappings
        # can be produced
        mappings = {
            "outer_inp_from_outer_out": {},
            "inner_inp_from_outer_out": {},
            "inner_out_from_outer_out": {},
            "inner_inp_from_outer_inp": {},
            "inner_out_from_outer_inp": {},
            "outer_out_from_outer_inp": {},
            "outer_inp_from_inner_inp": {},
            "inner_out_from_inner_inp": {},
            "outer_out_from_inner_inp": {},
            "outer_inp_from_inner_out": {},
            "inner_inp_from_inner_out": {},
            "outer_out_from_inner_out": {},
        }

        for (oinp, iinp, iout, oout) in zip(
            outer_input_indices,
            inner_input_indices,
            inner_output_indices,
            outer_output_indices,
        ):

            if oout != -1:
                mappings["outer_inp_from_outer_out"][oout] = oinp
                mappings["inner_inp_from_outer_out"][oout] = iinp
                mappings["inner_out_from_outer_out"][oout] = iout

            if oinp != -1:
                mappings["inner_inp_from_outer_inp"][oinp] = iinp
                mappings["inner_out_from_outer_inp"][oinp] = iout
                mappings["outer_out_from_outer_inp"][oinp] = oout

            for idx in iinp:
                mappings["outer_inp_from_inner_inp"][idx] = oinp
                mappings["inner_out_from_inner_inp"][idx] = iout
                mappings["outer_out_from_inner_inp"][idx] = oout

            for idx in iout:
                mappings["outer_inp_from_inner_out"][idx] = oinp
                mappings["inner_inp_from_inner_out"][idx] = iinp
                mappings["outer_out_from_inner_out"][idx] = oout

        return mappings

    def validate_inner_graph(self):
        """
        Perform some elementary validations on the inner graph to ensure
        that it is coherent.

        """

        # For every recurrent output, iterate over the associated inner
        # inputs and output and ensure that they have the same dtype
        nb_recurr_outputs = (
            self.info.n_mit_mot + self.info.n_mit_sot + self.info.n_sit_sot
        )
        var_mappings = self.get_oinp_iinp_iout_oout_mappings()

        for outer_oidx in range(nb_recurr_outputs):

            inner_iidxs = var_mappings["inner_inp_from_outer_out"][outer_oidx]
            inner_oidxs = var_mappings["inner_out_from_outer_out"][outer_oidx]

            for (inner_iidx, inner_oidx) in product(inner_iidxs, inner_oidxs):

                type_input = self.inner_inputs[inner_iidx].type
                type_output = self.inner_outputs[inner_oidx].type
                if (
                    # TODO: Use the `Type` interface for this
                    type_input.dtype != type_output.dtype
                    or type_input.broadcastable != type_output.broadcastable
                ):
                    raise TypeError(
                        "Inconsistency in the inner graph of "
                        f"scan '{self.name}' : an input and an output are "
                        "associated with the same recurrent state "
                        "and should have compatible types but have "
                        f"type '{type_input}' and '{type_output}' respectively."
                    )


class Scan(Op, ScanMethodsMixin, HasInnerGraph):
    r"""An `Op` implementing `for` and `while` loops.

    This `Op` has an "inner-graph" that represents the steps performed during
    each iteration (or in the body of its loops).  The vernacular for `Scan`
    uses the prefix "inner-" for things pertaining to the
    aforementioned inner-graph and "outer-" for node/`Apply`-level
    things.  There are inputs and outputs for both, and the "outer-graph" is
    the graph in which the `Scan`-using `Apply` node exists.

    The term "tap" refers to a relationship between the inputs and outputs of
    an inner-graph.  There are four types of taps and they characterize all the
    supported "connection" patterns between inner-graph inputs and outputs:

        - nit-sot (i.e. no inputs and a single output): A nit-sot is an output
        variable of the inner-graph that is not fed back as an input to the
        next iteration of the inner function.

        - sit-sot (i.e. a single input and a single output): A sit-sot is an output
        variable of the inner-graph that is fed back as an input to the next
        iteration of the inner-graph.

        - mit-sot (i.e. multiple inputs and a single output): A mit-sot is an
        output variable of the inner-graph that is fed back as an input to
        future iterations of the inner function (either multiple future
        iterations or a single one that isn't the immediate next one).

        - mit-mot (i.e. multiple inputs and multiple outputs): TODO

    """

    def __init__(
        self,
        inputs: List[Variable],
        outputs: List[Variable],
        info: ScanInfo,
        mode: Optional[Mode] = None,
        typeConstructor: Optional[TensorConstructorType] = None,
        truncate_gradient: int = -1,
        name: Optional[str] = None,
        as_while: bool = False,
        profile: Optional[Union[str, bool]] = None,
        allow_gc: bool = True,
        strict: bool = True,
    ):
        r"""

        Parameters
        ----------
        inputs
            Inputs of the inner function of `Scan`.
            These take the following general form:

                sequences + mit-mot-inputs + mit-sot-inputs + sit-sot-inputs + shared-inputs + non-sequences

            where each term is a list of `Variable`\s.

        outputs
            Outputs of the inner function of `Scan`.
            These take the following general form:

                mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots + shared-outputs [+ while-condition]

            where each term is a list of `Variable`\s.

        info
            A collection of information about the sequences and taps.
        mode
            The mode used to compile the inner-graph.
            If you prefer the computations of one step of `scan` to be done
            differently then the entire function, you can use this parameter to
            describe how the computations in this loop are done (see
            `aesara.function` for details about possible values and their meaning).
        typeConstructor
            Function that constructs a `TensorType` for the outputs.
        truncate_gradient
            `truncate_gradient` is the number of steps to use in truncated
            back-propagation through time (BPTT).  If you compute gradients through
            a `Scan` `Op`, they are computed using BPTT. By providing a different
            value then ``-1``, you choose to use truncated BPTT instead of classical
            BPTT, where you go for only `truncate_gradient` number of steps back in
            time.
        name
            When profiling `scan`, it is helpful to provide a name for any
            instance of `scan`.
            For example, the profiler will produce an overall profile of your code
            as well as profiles for the computation of one step of each instance of
            `Scan`. The `name` of the instance appears in those profiles and can
            greatly help to disambiguate information.
        profile
            If ``True`` or a non-empty string, a profile object will be created and
            attached to the inner graph of `Scan`. When `profile` is ``True``, the
            profiler results will use the name of the `Scan` instance, otherwise it
            will use the passed string.  The profiler only collects and prints
            information when running the inner graph with the `CVM` `Linker`.
        allow_gc
            Set the value of `allow_gc` for the internal graph of the `Scan`.  If
            set to ``None``, this will use the value of
            `aesara.config.scan__allow_gc`.

            The full `Scan` behavior related to allocation is determined by this
            value and the flag `aesara.config.allow_gc`. If the flag
            `allow_gc` is ``True`` (default) and this `allow_gc` is ``False``
            (default), then we let `Scan` allocate all intermediate memory
            on the first iteration, and they are not garbage collected
            after that first iteration; this is determined by `allow_gc`. This can
            speed up allocation of the subsequent iterations. All those temporary
            allocations are freed at the end of all iterations; this is what the
            flag `aesara.config.allow_gc` means.
        strict
            If ``True``, all the shared variables used in the inner-graph must be provided.

        """
        self.fgraph, shared_inputs, _, _ = construct_nominal_fgraph(inputs, outputs)

        # The shared variables should have been removed, so, if there are
        # any, it's because the user didn't specify an input.
        if shared_inputs:
            raise MissingInputError(f"Scan is missing inputs: {shared_inputs}")

        self.info = info
        self.truncate_gradient = truncate_gradient
        self.name = name
        self.profile = profile
        self.allow_gc = allow_gc
        self.strict = strict

        # Clone mode_instance, altering "allow_gc" for the linker,
        # and adding a message if we profile
        if self.name:
            message = f"{self.name} sub profile"
        else:
            message = "Scan sub profile"

        self.mode = get_default_mode() if mode is None else mode
        self.mode_instance = get_mode(self.mode).clone(
            link_kwargs=dict(allow_gc=self.allow_gc), message=message
        )

        # build a list of output types for any Apply node using this op.
        self.output_types = []

        def tensorConstructor(shape, dtype):
            return TensorType(dtype=dtype, shape=shape)

        if typeConstructor is None:
            typeConstructor = tensorConstructor

        idx = 0
        jdx = 0
        while idx < info.n_mit_mot_outs:
            # Not that for mit_mot there are several output slices per
            # output sequence
            o = self.fgraph.outputs[idx]
            self.output_types.append(
                # TODO: What can we actually say about the shape of this
                # added dimension?
                typeConstructor((None,) + o.type.shape, o.type.dtype)
            )

            idx += len(info.mit_mot_out_slices[jdx])
            jdx += 1

        # mit_sot / sit_sot / nit_sot
        end = idx + info.n_mit_sot + info.n_sit_sot + info.n_nit_sot

        for o in self.fgraph.outputs[idx:end]:
            self.output_types.append(
                # TODO: What can we actually say about the shape of this
                # added dimension?
                typeConstructor((None,) + o.type.shape, o.type.dtype)
            )

        # shared outputs + possibly the ending condition
        for o in self.fgraph.outputs[end:]:
            self.output_types.append(o.type)

        if info.as_while:
            self.output_types = self.output_types[:-1]

        if not hasattr(self, "name") or self.name is None:
            self.name = "scan_fn"

        # Pre-computing some values to speed up perform
        self.mintaps = [
            min(x)
            for x in chain(
                info.mit_mot_in_slices, info.mit_sot_in_slices, info.sit_sot_in_slices
            )
        ]
        self.mintaps += [0 for x in range(info.n_nit_sot)]
        self.seqs_arg_offset = 1 + info.n_seqs
        self.shared_arg_offset = (
            self.seqs_arg_offset + info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
        )
        self.nit_sot_arg_offset = self.shared_arg_offset + info.n_shared_outs
        # XXX: This doesn't include `info.n_nit_sot`s, so it's really a count
        # of the number of outputs generated by taps with inputs
        self.n_outs = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
        self.n_tap_outs = info.n_mit_mot + info.n_mit_sot

        # TODO: These can be moved to thunk/function compilation
        (
            _,
            self.mitmots_preallocated,
        ) = self._mitmot_preallocations()

        self.n_outer_inputs = info.n_outer_inputs
        self.n_outer_outputs = info.n_outer_outputs

        _ = self.prepare_fgraph(self.fgraph)

        if any(node.op.destroy_map for node in self.fgraph.apply_nodes):
            raise InconsistencyError(
                "Inner-graphs must not contain in-place operations."
            )

        self._cmodule_key = CLinker().cmodule_key_variables(
            self.inner_inputs, self.inner_outputs, []
        )
        self._hash_inner_graph = hash(self._cmodule_key)

    def _mitmot_preallocations(self):
        if config.scan__allow_output_prealloc:
            preallocated_mitmot_outs = []

            info = self.info
            input_idx = info.n_seqs
            for mitmot_idx in range(info.n_mit_mot):
                for inp_tap in info.mit_mot_in_slices[mitmot_idx]:
                    if inp_tap in info.mit_mot_out_slices[mitmot_idx]:
                        # Figure out the index of the corresponding output
                        output_idx = sum(
                            len(m) for m in info.mit_mot_out_slices[:mitmot_idx]
                        )
                        output_idx += info.mit_mot_out_slices[mitmot_idx].index(inp_tap)
                        preallocated_mitmot_outs.append(output_idx)

                    input_idx += 1

            preallocated_mitmot_outs.sort()

        else:
            # Output preallocation is not activated. Mark every mitmot output
            # tap as not being preallocated
            preallocated_mitmot_outs = []

        # Store the list of mitmot output taps that have been altered so they
        # can be preallocated
        mitmots_preallocated = [
            i in preallocated_mitmot_outs for i in range(info.n_mit_mot_outs)
        ]

        return preallocated_mitmot_outs, mitmots_preallocated

    def __setstate__(self, d):
        self.__dict__.update(d)
        # Ensure that the graph associated with the inner function is valid.
        self.validate_inner_graph()

    def make_node(self, *inputs):
        """
        The `inputs` to this method take the following form:

            sequences +
            mit-mot-inputs + mit-sot-inputs + sit-sot-inputs +
            shared-inputs +
            nit-sots +
            non-sequences

        Note that some ``non-sequences`` can also be shared variables, and that
        ``nit-sots`` variables are the lengths of each nit-sot output, because
        nit-sots have no input connections (by definition).  Also, don't forget
        that mit-[s|m]ots each have a distinct number of inputs and/or outputs.

        The (outer-)inputs in the :class:`Apply` nodes created by this method
        take the following concatenative form:

            [n_steps] +
            sequences +
            mit-mot-inputs + mit-sot-inputs + sit-sot-inputs +
            shared-inputs +
            nit-sots +
            non-sequences

        The (outer-)outputs take the following form:

            mit-mot-outputs + mit-sot-outputs + sit-sot-outputs +
            nit-sots +
            shared-outputs

        These outer-outputs essentially follow the same form as their
        corresponding inner-outputs, excluding the final "while" condition
        term.

        """
        if not all(isinstance(i, Variable) for i in inputs):
            raise TypeError("Inputs must be `Variable` instances")

        # Check that the number of inputs to the Scan node corresponds to
        # the number of inputs of the inner function of scan
        n_outer_ins = len(inputs) - len(self.outer_nitsot(inputs)) - 1
        n_inner_ins = (
            len(self.inner_seqs(self.inner_inputs))
            + len(self.info.mit_mot_in_slices)
            + len(self.info.mit_sot_in_slices)
            + len(self.inner_sitsot(self.inner_inputs))
            + len(self.inner_shared(self.inner_inputs))
            + len(self.inner_non_seqs(self.inner_inputs))
        )

        if n_outer_ins != n_inner_ins:
            raise ValueError(
                "The number of inputs given to the inner function of scan"
                " does not match the number of inputs given to scan."
            )

        # Force the inputs to be on the CPU
        new_inputs = [as_tensor_variable(inputs[0])]

        # Check if input sequences and variables representing a slice of
        # them have the same dtype
        argoffset = 0
        for inner_seq, outer_seq in zip(
            self.inner_seqs(self.inner_inputs), self.outer_seqs(inputs)
        ):
            check_broadcast(outer_seq, inner_seq)
            new_inputs.append(copy_var_format(outer_seq, as_var=inner_seq))

        argoffset += len(self.outer_seqs(inputs))
        # Check that this 3 things have the same dtype for mit_mot:
        #   - initial state of the output
        #   - variable representing an input slice of the output
        #   - variable representing an output slice of the output
        ipos = 0
        opos = 0
        inner_mitmot = self.inner_mitmot(self.inner_inputs)
        inner_mitmot_outs = self.inner_mitmot_outs(self.inner_outputs)
        for idx, (itaps, otaps, _outer_mitmot) in enumerate(
            zip(
                self.info.mit_mot_in_slices,
                self.info.mit_mot_out_slices[: self.info.n_mit_mot],
                self.outer_mitmot(inputs),
            )
        ):
            outer_mitmot = copy_var_format(_outer_mitmot, as_var=inner_mitmot[ipos])
            new_inputs.append(outer_mitmot)
            for k in range(len(itaps)):
                if (
                    inner_mitmot[ipos + k].type.dtype != outer_mitmot.type.dtype
                    or inner_mitmot[ipos + k].ndim != outer_mitmot.ndim - 1
                ):
                    raise ValueError(
                        err_msg1
                        % (
                            "initial state (`outputs_info` in scan nomenclature) ",
                            str(outer_mitmot),
                            argoffset + idx,
                            outer_mitmot.type.dtype,
                            outer_mitmot.type.ndim,
                            str(inner_mitmot[ipos + k]),
                            inner_mitmot[ipos + k].type.dtype,
                            inner_mitmot[ipos + k].type.ndim,
                        )
                    )
            ipos += len(itaps)
            for k in range(len(otaps)):
                if inner_mitmot_outs[opos + k].type.dtype != outer_mitmot.type.dtype:
                    raise ValueError(
                        err_msg2
                        % (
                            str(outer_mitmot),
                            argoffset + idx,
                            outer_mitmot.type.dtype,
                            inner_mitmot_outs[opos + k].type.dtype,
                        )
                    )
                if inner_mitmot_outs[opos + k].ndim != outer_mitmot.ndim - 1:
                    raise ValueError(
                        err_msg3
                        % (
                            str(outer_mitmot),
                            argoffset + idx,
                            outer_mitmot.ndim,
                            inner_mitmot_outs[opos + k].ndim,
                        )
                    )
            opos += len(otaps)
        argoffset += len(self.outer_mitmot(inputs))
        # Same checks as above but for outputs of type mit_sot
        ipos = 0
        inner_mitsots = self.inner_mitsot(self.inner_inputs)
        for idx, (itaps, _outer_mitsot, inner_mitsot_out) in enumerate(
            zip(
                self.info.mit_sot_in_slices,
                self.outer_mitsot(inputs),
                self.inner_mitsot_outs(self.inner_outputs),
            )
        ):
            outer_mitsot = copy_var_format(_outer_mitsot, as_var=inner_mitsots[ipos])
            new_inputs.append(outer_mitsot)

            for k in range(len(itaps)):
                if (
                    inner_mitsots[ipos + k].type.dtype != outer_mitsot.type.dtype
                    or inner_mitsots[ipos + k].ndim != outer_mitsot.ndim - 1
                ):
                    raise ValueError(
                        err_msg1
                        % (
                            "initial state (outputs_info" " in scan nomenclature) ",
                            str(outer_mitsot),
                            argoffset + idx,
                            outer_mitsot.type.dtype,
                            outer_mitsot.type.ndim,
                            str(inner_mitsots[ipos + k]),
                            inner_mitsots[ipos + k].type.dtype,
                            inner_mitsots[ipos + k].type.ndim,
                        )
                    )
            ipos += len(itaps)
            if inner_mitsot_out.type.dtype != outer_mitsot.type.dtype:
                raise ValueError(
                    err_msg2
                    % (
                        str(outer_mitsot),
                        argoffset + idx,
                        outer_mitsot.type.dtype,
                        inner_mitsot_out.type.dtype,
                    )
                )
            if inner_mitsot_out.ndim != outer_mitsot.ndim - 1:
                raise ValueError(
                    err_msg3
                    % (
                        str(outer_mitsot),
                        argoffset + idx,
                        outer_mitsot.ndim,
                        inner_mitsot_out.ndim,
                    )
                )

        argoffset += len(self.outer_mitsot(inputs))
        # Same checks as above but for outputs of type sit_sot
        for idx, (inner_sitsot, _outer_sitsot, inner_sitsot_out) in enumerate(
            zip(
                self.inner_sitsot(self.inner_inputs),
                self.outer_sitsot(inputs),
                self.inner_sitsot_outs(self.inner_outputs),
            )
        ):
            outer_sitsot = copy_var_format(_outer_sitsot, as_var=inner_sitsot)
            new_inputs.append(outer_sitsot)
            if inner_sitsot.ndim != outer_sitsot.ndim - 1:
                raise ValueError(
                    err_msg1
                    % (
                        "initial state (outputs_info" " in scan nomenclature) ",
                        str(outer_sitsot),
                        argoffset + idx,
                        outer_sitsot.type.dtype,
                        outer_sitsot.type.ndim,
                        str(inner_sitsot),
                        inner_sitsot.type.dtype,
                        inner_sitsot.type.ndim,
                    )
                )
            if inner_sitsot_out.type.dtype != outer_sitsot.type.dtype:
                raise ValueError(
                    err_msg2
                    % (
                        str(outer_sitsot),
                        argoffset + idx,
                        outer_sitsot.type.dtype,
                        inner_sitsot_out.type.dtype,
                    )
                )
            if inner_sitsot_out.ndim != outer_sitsot.ndim - 1:
                raise ValueError(
                    err_msg3
                    % (
                        str(outer_sitsot),
                        argoffset + idx,
                        outer_sitsot.type.ndim,
                        inner_sitsot_out.type.ndim,
                    )
                )

        argoffset += len(self.outer_sitsot(inputs))
        # Check that the shared variable and their update rule have the same
        # dtype. Maybe even same type ?!
        for idx, (inner_shared, inner_shared_out, _outer_shared) in enumerate(
            zip(
                self.inner_shared(self.inner_inputs),
                self.inner_shared_outs(self.inner_outputs),
                self.outer_shared(inputs),
            )
        ):
            outer_shared = copy_var_format(_outer_shared, as_var=inner_shared)
            new_inputs.append(outer_shared)
            if (
                hasattr(outer_shared, "dtype")
                and outer_shared.dtype != inner_shared_out.dtype
            ):
                raise ValueError(
                    err_msg2
                    % (
                        str(outer_shared),
                        idx + argoffset,
                        outer_shared.dtype,
                        inner_shared_out.dtype,
                    )
                )
            if (
                hasattr(outer_shared, "dtype")
                and outer_shared.ndim != inner_shared_out.ndim
            ):
                raise ValueError(
                    err_msg3
                    % (
                        str(outer_shared),
                        idx + argoffset,
                        outer_shared.ndim,
                        inner_shared_out.ndim,
                    )
                )

            if hasattr(outer_shared, "dtype") and (
                outer_shared.dtype != inner_shared.dtype
                or outer_shared.ndim != inner_shared.ndim
            ):
                raise ValueError(
                    err_msg1
                    % (
                        "initial state (outputs_info" " in scan nomenclature) ",
                        str(outer_shared),
                        argoffset + idx,
                        outer_shared.dtype,
                        outer_shared.ndim,
                        str(inner_shared),
                        inner_shared.dtype,
                        inner_shared.ndim,
                    )
                )
        # We do not need to call `copy_var_format` on outer_nisot arguments.
        # outer_nitsot stands for no input tap single output tap. This means
        # these are states that do not feed anything back in the recurrent
        # computation, and hence they do not have an initial state. The scan
        # node however receives an input for each such argument, the input
        # in this case is just a int saying how many steps of this output we
        # need to store. This input does not have the same dtype, nor is it the same
        # type of tensor as the output, it is always a scalar int.
        new_inputs += [as_tensor_variable(ons) for ons in self.outer_nitsot(inputs)]
        for inner_nonseq, _outer_nonseq in zip(
            self.inner_non_seqs(self.inner_inputs), self.outer_non_seqs(inputs)
        ):
            outer_nonseq = copy_var_format(_outer_nonseq, as_var=inner_nonseq)
            new_inputs.append(outer_nonseq)
            if not outer_nonseq.type.in_same_class(inner_nonseq.type):
                raise ValueError(
                    (
                        f"Argument {outer_nonseq} given to the scan node is not"
                        f" compatible with its corresponding loop function variable {inner_nonseq}"
                    )
                )

        for outer_nitsot in self.outer_nitsot(inputs):
            # For every nit_sot input we get as input a int/uint that
            # depicts the size in memory for that sequence. This feature is
            # used by truncated BPTT and by scan space optimization
            if (
                str(outer_nitsot.type.dtype) not in integer_dtypes
                or outer_nitsot.ndim != 0
            ):
                raise ValueError(f"A scalar int is required for output {outer_nitsot}")

        assert len(new_inputs) == len(inputs)

        # The vector_seqs and vector_outs are just a workaround
        # strange NumPy behavior: vector_ndarray[int] return a NumPy
        # scalar and not a NumPy ndarray of 0 dimensions.
        def is_cpu_vector(s):
            return isinstance(s.type, TensorType) and s.ndim == 1

        self.vector_seqs = [
            is_cpu_vector(seq) for seq in new_inputs[1 : 1 + self.info.n_seqs]
        ]
        self.vector_outs = [
            is_cpu_vector(arg)
            for arg in new_inputs[
                1 + self.info.n_seqs : (1 + self.info.n_seqs + self.n_outs)
            ]
        ]
        self.vector_outs += [
            isinstance(t.type, TensorType) and t.ndim == 0
            for t in self.outer_nitsot_outs(self.inner_outputs)
        ]

        outputs = [t() for t in self.output_types]

        assert self.n_outer_inputs == len(new_inputs), (
            self.n_outer_inputs,
            len(new_inputs),
        )
        assert self.n_outer_outputs == len(outputs), (
            self.n_outer_outputs,
            len(outputs),
        )

        apply_node = Apply(self, new_inputs, outputs)
        return apply_node

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.info != other.info:
            return False

        if self.profile != other.profile:
            return False

        if self.truncate_gradient != other.truncate_gradient:
            return False

        if self.name != other.name:
            return False

        if self.allow_gc != other.allow_gc:
            return False

        # Compare inner graphs
        # TODO: Use `self.inner_fgraph == other.inner_fgraph`
        if len(self.inner_inputs) != len(other.inner_inputs):
            return False

        if len(self.inner_outputs) != len(other.inner_outputs):
            return False

        for self_in, other_in in zip(self.inner_inputs, other.inner_inputs):
            if self_in.type != other_in.type:
                return False

        return equal_computations(
            self.inner_outputs,
            other.inner_outputs,
            self.inner_inputs,
            other.inner_inputs,
        )

    def __str__(self):
        device_str = "cpu"
        if self.info.as_while:
            name = "do_while"
        else:
            name = "for"
        aux_txt = "%s"
        if len(self.destroy_map.keys()) > 0:
            # Check if all outputs are inplace
            if sorted(self.destroy_map.keys()) == sorted(
                range(self.info.n_mit_mot + self.info.n_mit_sot + self.info.n_sit_sot)
            ):
                aux_txt += "all_inplace,%s,%s}"
            else:
                aux_txt += "{inplace{"
                for k in self.destroy_map.keys():
                    aux_txt += str(k) + ","
                aux_txt += "},%s,%s}"
        else:
            aux_txt += "{%s,%s}"
        aux_txt = aux_txt % (name, device_str, str(self.name))
        return aux_txt

    def __hash__(self):
        return hash(
            (
                type(self),
                self._hash_inner_graph,
                self.info,
                self.profile,
                self.truncate_gradient,
                self.name,
                self.allow_gc,
            )
        )

    def prepare_fgraph(self, fgraph):
        """Update and wrap `fgraph`'s inputs and outputs in preparation for compilation."""

        info = self.info
        slices = info.n_mit_mot_outs + info.n_mit_sot + info.n_sit_sot + info.n_nit_sot

        # Setting `fgraph.update_mapping` will indicate to the `Function`
        # construction pipeline that it needn't append the updates to the
        # `FunctionGraph` outputs itself, because they're already in the given
        # `FunctionGraph`'s outputs.  This also prevents us from needing to
        # remove those outputs here just to compensate for an overly rigid
        # `Function` pipeline.
        update_mapping = {}

        preallocated_mitmot_outs = []

        if config.scan__allow_output_prealloc:

            # Go through the mitmots. Whenever a mitmot has a tap both as an
            # input and an output, wrap the input such that the corresponding
            # output variable becomes an update to be performed on it, possibly
            # inplace at the end of the functions's execution.
            wrapped_inputs = [In(x, borrow=False) for x in fgraph.inputs[: info.n_seqs]]

            input_idx = info.n_seqs
            for mitmot_idx in range(info.n_mit_mot):
                for inp_tap in info.mit_mot_in_slices[mitmot_idx]:
                    if inp_tap in info.mit_mot_out_slices[mitmot_idx]:
                        inp = fgraph.inputs[input_idx]

                        # Figure out the index of the corresponding output
                        output_idx = sum(
                            len(m) for m in info.mit_mot_out_slices[:mitmot_idx]
                        )
                        output_idx += info.mit_mot_out_slices[mitmot_idx].index(inp_tap)

                        preallocated_mitmot_outs.append(output_idx)

                        # Make it so that the input is automatically updated to
                        # the output value, possibly inplace, at the end of the
                        # function execution. Also, since an update is defined,
                        # a default value must also be (this is verified by
                        # DebugMode).
                        # TODO FIXME: Why do we need a "default value" here?
                        # This sounds like a serious design issue.
                        default_shape = tuple(
                            s if s is not None else 0 for s in inp.type.shape
                        )
                        default_val = np.empty(default_shape, dtype=inp.type.dtype)
                        wrapped_inp = In(
                            variable=inp,
                            value=default_val,
                            update=fgraph.outputs[output_idx],
                        )
                        update_mapping[output_idx] = input_idx
                        wrapped_inputs.append(wrapped_inp)
                    else:
                        wrapped_inputs.append(
                            In(fgraph.inputs[input_idx], borrow=False)
                        )
                    input_idx += 1

            # Wrap the inputs not associated to mitmots and wrap the remaining
            # outputs
            wrapped_inputs += [In(x, borrow=False) for x in fgraph.inputs[input_idx:]]
            wrapped_outputs = [Out(x, borrow=True) for x in fgraph.outputs[:slices]]
            wrapped_outputs += fgraph.outputs[slices:]

            protected_outs = tuple(
                i
                for i in range(
                    info.n_mit_mot_outs
                    + info.n_mit_sot
                    + info.n_sit_sot
                    + info.n_nit_sot
                )
                if i not in preallocated_mitmot_outs
            )
            fgraph.attach_feature(NoOutputFromInplace(protected_outs))

        else:
            wrapped_inputs = [In(x, borrow=True) for x in fgraph.inputs]
            wrapped_outputs = [Out(x, borrow=False) for x in fgraph.outputs[:slices]]
            wrapped_outputs += fgraph.outputs[slices:]

        fgraph.update_mapping = update_mapping

        from aesara.compile.function.types import Supervisor
        from aesara.graph.destroyhandler import DestroyHandler

        for node in fgraph.apply_nodes:
            if node.op.destroy_map:
                fgraph.attach_feature(DestroyHandler())
                break

        fgraph.attach_feature(
            Supervisor(
                inp
                for spec, inp in zip(wrapped_inputs, fgraph.inputs)
                if not (
                    getattr(spec, "mutable", None)
                    or (hasattr(fgraph, "destroyers") and fgraph.has_destroyers([inp]))
                )
            )
        )

        return wrapped_inputs, wrapped_outputs

    @property
    def fn(self):
        """Lazily compile the inner function graph."""
        if getattr(self, "_fn", None) is not None:
            return self._fn

        wrapped_inputs, wrapped_outputs = self.prepare_fgraph(self.fgraph)

        profile = None
        if config.profile or (
            isinstance(self.profile, (str, bool, (int,))) and self.profile
        ):
            if isinstance(self.profile, str):
                profile = ScanProfileStats(name=self.profile)
            else:
                profile = ScanProfileStats(name=self.name)
        elif self.profile:
            profile = self.profile

        self._fn = pfunc(
            wrapped_inputs,
            wrapped_outputs,
            mode=self.mode_instance,
            accept_inplace=False,
            profile=profile,
            on_unused_input="ignore",
            fgraph=self.fgraph,
        )

        return self._fn

    @property
    def inner_inputs(self):
        return self.fgraph.inputs

    @property
    def inner_outputs(self):
        return self.fgraph.outputs

    def clone(self) -> "Scan":
        res = copy(self)
        res.fgraph = res.fgraph.clone()
        return res

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        """

        Parameters
        ----------
        node
            Something previously returned by self.make_node.
        storage_map
            dict variable -> one-element-list where a computed
            value for this variable may be found.
        compute_map
            dict variable -> one-element-list where a boolean
            value will be found. The boolean indicates whether the
            variable's storage_map container contains a valid value (True)
            or if it has not been computed yet (False).
        no_recycling
            List of variables for which it is forbidden to reuse memory
            allocated by a previous call.
        impl
            Use 'py' if we want python execution.
        Notes
        -----
        If the thunk consults the storage_map on every call, it is safe
        for it to ignore the no_recycling argument, because elements of the
        no_recycling list will have a value of None in the storage map. If
        the thunk can potentially cache return values (like CLinker does),
        then it must not do so for variables in the no_recycling list.

        """

        # Before building the thunk, validate that the inner graph is
        # coherent
        self.validate_inner_graph()

        # Setting up all my variables in what I believe is a more Cython
        # friendly form

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        # Analyse the compile inner function to determine which inputs and
        # outputs are on the gpu and speed up some checks during the execution
        outs_is_tensor = [
            isinstance(out, TensorVariable) for out in self.fn.maker.fgraph.outputs
        ]

        try:
            if impl == "py":
                raise MissingGXX

            from . import scan_perform_ext

            cython_mintaps = np.asarray(self.mintaps, dtype="int32")

            n_outs = self.info.n_mit_mot + self.info.n_mit_sot + self.info.n_sit_sot
            cython_pos = np.zeros(n_outs + self.info.n_nit_sot, dtype=np.uint32)
            cython_store_steps = np.zeros(n_outs + self.info.n_nit_sot, dtype=np.uint32)

            tap_array = (
                self.info.mit_mot_in_slices
                + self.info.mit_sot_in_slices
                + self.info.sit_sot_in_slices
            )
            tap_array_len = np.array([len(x) for x in tap_array], dtype=np.uint32)

            cython_vector_seqs = np.asarray(self.vector_seqs, dtype=bool)
            cython_vector_outs = np.asarray(self.vector_outs, dtype=bool)
            cython_mitmots_preallocated = np.asarray(
                self.mitmots_preallocated, dtype=bool
            )
            cython_outs_is_tensor = np.asarray(outs_is_tensor, dtype=bool)

            if self.destroy_map:
                cython_destroy_map = [
                    x in self.destroy_map for x in range(len(node.outputs))
                ]
            else:
                cython_destroy_map = [False for x in range(len(node.outputs))]

            cython_destroy_map = np.asarray(cython_destroy_map, dtype=bool)

            inner_input_storage = [s.storage for s in self.fn.input_storage]
            inner_output_storage = [s.storage for s in self.fn.output_storage]

            outer_output_dtypes = tuple(
                getattr(out, "dtype", object) for out in node.outputs
            )

            outer_output_ndims = np.array(
                [getattr(out, "ndim", 0) for out in node.outputs], dtype=np.uint32
            )

            # The input index for each mit-mot output
            mit_mot_out_to_tap_idx = ()
            for j in range(self.info.n_mit_mot):
                for k in self.info.mit_mot_out_slices[j]:
                    mit_mot_out_to_tap_idx += (tap_array[j].index(k),)
            mit_mot_out_to_tap_idx = np.asarray(mit_mot_out_to_tap_idx, dtype=np.uint32)

            from aesara.scan.utils import InnerFunctionError

            # TODO: Extract `Capsule` object and use that
            # c_thunk = getattr(self.fn.vm.thunks[0], "cthunk", None)
            # if len(self.fn.vm.thunks) == 1 and c_thunk:
            #     thunk_capsule = c_thunk.cthunk
            #     # We need to perform the following after calling
            #     # the thunk function:
            #     # for o in node.outputs:
            #     #     compute_map[o][0] = True

            def p(node, inputs, outputs):

                t0_call = time.perf_counter()

                try:
                    t_fn, n_steps = scan_perform_ext.perform(
                        self.info.n_shared_outs,
                        self.info.n_mit_mot_outs,
                        self.info.n_seqs,
                        self.info.n_mit_mot,
                        self.info.n_mit_sot,
                        self.info.n_sit_sot,
                        self.info.n_nit_sot,
                        self.info.as_while,
                        cython_mintaps,
                        cython_pos,
                        cython_store_steps,
                        self.info.mit_mot_in_slices
                        + self.info.mit_sot_in_slices
                        + self.info.sit_sot_in_slices,
                        tap_array_len,
                        cython_vector_seqs,
                        cython_vector_outs,
                        self.info.mit_mot_out_slices,
                        cython_mitmots_preallocated,
                        mit_mot_out_to_tap_idx,
                        cython_outs_is_tensor,
                        inner_input_storage,
                        inner_output_storage,
                        cython_destroy_map,
                        inputs,
                        outputs,
                        outer_output_dtypes,
                        outer_output_ndims,
                        self.fn.vm,
                    )
                except InnerFunctionError as exc:
                    exc_type = type(exc.args[0])
                    exc_value = exc.args[0]
                    exc_trace = exc.args[1]

                    if hasattr(self.fn.vm, "position_of_error") and hasattr(
                        self.fn.vm, "thunks"
                    ):
                        raise_with_op(
                            self.fn.maker.fgraph,
                            self.fn.vm.nodes[self.fn.vm.position_of_error],
                            self.fn.vm.thunks[self.fn.vm.position_of_error],
                            exc_info=(exc_type, exc_value, exc_trace),
                        )
                    else:
                        raise exc_value.with_traceback(exc_trace)

                t_call = time.perf_counter() - t0_call

                if hasattr(self.fn.maker, "profile"):
                    profile = self.fn.maker.profile
                    if type(profile) is not bool and profile:
                        profile.vm_call_time += t_fn
                        profile.callcount += 1
                        profile.nbsteps += n_steps
                        profile.call_time += t_call
                        if hasattr(self.fn.vm, "update_profile"):
                            self.fn.vm.update_profile(profile)

        except (ImportError, MissingGXX):
            p = self.perform

        # default arguments are stored in the closure of `rval`

        # Big ugly hack since we can't get the real value of allow_gc
        # for the englobing function.
        allow_gc = config.allow_gc and not self.allow_gc

        def rval(
            p=p, i=node_input_storage, o=node_output_storage, n=node, allow_gc=allow_gc
        ):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            if allow_gc:
                self.fn.free()
            return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def perform(self, node, inputs, output_storage, params=None):
        """Compute the scan operation in Python.

        The `inputs` are packed like this:

            n_steps

            X sequence inputs x_1, x_2, ... x_<self.info.n_seqs>

            Y initial states (u_1, u_2, ... u_<self.n_outs>) for our
            outputs. Each must have appropriate length (T_1, T_2, ..., T_Y).

            W other inputs w_1, w_2, ... w_W

        There are at least ``1 + self.info.n_seqs + self.n_outs`` inputs, and the
        ones above this number are passed to the scanned function as
        non-sequential inputs.

        The outputs are more straightforward:

            Y sequence outputs y_1, y_2, ... y_<self.n_outs>

        """
        info = self.info
        # 1. Unzip the number of steps and sequences.
        t0_call = time.perf_counter()
        t_fn = 0
        n_steps = inputs[0]
        seqs = []
        if n_steps < 0:
            # History, in the past, this was used for backward
            # scan. Now we reverse the inputs outside of scan.
            raise IndexError(
                f"Scan was asked to run for negative number of step {n_steps}"
            )
        else:
            for idx, seq in enumerate(inputs[1 : self.seqs_arg_offset]):
                if seq.shape[0] < n_steps:
                    raise ValueError(
                        f"Sequence {idx} has shape {seq.shape} "
                        f"but the Scan's required number of steps is {n_steps}"
                    )
                seqs.append(seq)

        # 2. Allocate memory for the outputs. Construct the list:

        # The length of each output
        store_steps = [
            arg.shape[0]
            for arg in inputs[self.seqs_arg_offset : self.shared_arg_offset]
        ]
        store_steps += [
            arg
            for arg in inputs[
                self.nit_sot_arg_offset : self.nit_sot_arg_offset + info.n_nit_sot
            ]
        ]

        # 2.1 Create storage space for outputs
        for idx in range(self.n_outs):
            if idx in self.destroy_map:
                # ^ Case 1. Outputs should be computed inplace of their
                # initial state
                output_storage[idx][0] = inputs[self.seqs_arg_offset + idx]
            elif (
                output_storage[idx][0] is not None
                and output_storage[idx][0].shape[1:]
                == inputs[self.seqs_arg_offset + idx].shape[1:]
                and output_storage[idx][0].shape[0] >= store_steps[idx]
            ):
                # Put in the values of the initial state
                output_storage[idx][0] = output_storage[idx][0][: store_steps[idx]]
                if idx > info.n_mit_mot:
                    l = -self.mintaps[idx]
                    output_storage[idx][0][:l] = inputs[self.seqs_arg_offset + idx][:l]
                else:
                    output_storage[idx][0][:] = inputs[self.seqs_arg_offset + idx]
            else:
                output_storage[idx][0] = inputs[self.seqs_arg_offset + idx].copy()

        if n_steps == 0:
            for idx in range(self.n_outs, self.n_outs + info.n_nit_sot):
                out_var = node.outputs[idx]
                if isinstance(out_var, TensorVariable):
                    output_storage[idx][0] = np.empty(
                        (0,) * out_var.type.ndim, dtype=out_var.type.dtype
                    )
                else:
                    output_storage[idx][0] = None
            return

        # The current position of each output
        pos = [
            (-self.mintaps[idx]) % store_steps[idx]
            for idx in range(self.n_outs + info.n_nit_sot)
        ]

        offset = self.nit_sot_arg_offset + info.n_nit_sot
        other_args = inputs[offset:]
        inner_input_storage = self.fn.input_storage
        nb_mitmot_in = sum(map(len, info.mit_mot_in_slices))
        old_mitmot_input_storage = [None] * nb_mitmot_in
        old_mitmot_input_data = [None] * nb_mitmot_in
        inner_output_storage = self.fn.output_storage
        old_inner_output_storage = [None] * len(inner_output_storage)
        old_inner_output_data = [None] * len(inner_output_storage)
        vm = self.fn.vm
        offset = (
            info.n_seqs
            + sum(
                len(x)
                for x in chain(
                    info.mit_mot_in_slices,
                    info.mit_sot_in_slices,
                    info.sit_sot_in_slices,
                )
            )
            + info.n_shared_outs
        )
        for idx in range(len(other_args)):
            inner_input_storage[idx + offset].storage[0] = other_args[idx]

        i = 0
        cond = True
        # ############# THE MAIN LOOP ##############
        # for i in range(n_steps):
        while (i < n_steps) and cond:
            # sequences over which scan iterates
            # 3. collect input slices
            for idx in range(info.n_seqs):
                if self.vector_seqs[idx]:
                    inner_input_storage[idx].storage[0] = seqs[idx][i : i + 1].reshape(
                        ()
                    )
                else:
                    inner_input_storage[idx].storage[0] = seqs[idx][i]

            offset = info.n_seqs
            for idx, taps in enumerate(
                chain(
                    info.mit_mot_in_slices,
                    info.mit_sot_in_slices,
                    info.sit_sot_in_slices,
                )
            ):
                if self.vector_outs[idx]:
                    for t in taps:
                        _idx = (pos[idx] + t) % store_steps[idx]
                        inner_input_storage[offset].storage[0] = output_storage[idx][0][
                            _idx : _idx + 1
                        ].reshape(())
                        offset += 1
                else:
                    for t in taps:
                        _idx = (pos[idx] + t) % store_steps[idx]
                        inner_input_storage[offset].storage[0] = output_storage[idx][0][
                            _idx
                        ]
                        offset += 1

            a_offset = self.shared_arg_offset
            o_offset = self.n_outs + info.n_nit_sot
            if i == 0:
                for j in range(info.n_shared_outs):
                    inner_input_storage[offset].storage[0] = inputs[a_offset + j]
                    offset += 1
            else:
                for j in range(info.n_shared_outs):
                    inner_input_storage[offset].storage[0] = output_storage[
                        o_offset + j
                    ][0]
                    offset += 1

            # 4. collecting slices where the output should be stored

            # 4.1. Collect slices for mitmots
            offset = 0
            for idx in range(info.n_mit_mot_outs):
                if not self.mitmots_preallocated[idx]:
                    inner_output_storage[offset].storage[0] = None
                offset += 1

            # 4.2. Collect slices for mitsots, sitsots and nitsots
            if i != 0:
                for idx in range(self.n_outs + info.n_nit_sot - info.n_mit_mot):
                    if (
                        store_steps[idx + info.n_mit_mot] == 1
                        or self.vector_outs[idx + info.n_mit_mot]
                    ):
                        inner_output_storage[idx + offset].storage[0] = None
                    else:
                        _pos0 = idx + info.n_mit_mot
                        inner_output_storage[idx + offset].storage[0] = output_storage[
                            _pos0
                        ][0][pos[_pos0]]
            else:
                for idx in range(self.n_outs + info.n_nit_sot - info.n_mit_mot):
                    inner_output_storage[idx + offset].storage[0] = None

            # 4.3. Collect slices for shared outputs
            offset += self.n_outs + info.n_nit_sot - info.n_mit_mot
            for idx in range(info.n_shared_outs):
                inner_output_storage[idx + offset].storage[0] = None

            # 4.4. If there is a condition add it to the mix
            if info.as_while:
                pdx = offset + info.n_shared_outs
                inner_output_storage[pdx].storage[0] = None

            # 4.5. Keep a reference to the variables (ndarrays,
            # etc) currently in the output_storage to be able to compare them
            # with the actual outputs of the inner function after its
            # execution. Also keep pointers to their data to be able to detect
            # cases where outputs reused the allocated object but alter the
            # memory region they refer to.
            for idx in range(len(inner_output_storage)):

                var = inner_output_storage[idx].storage[0]
                old_inner_output_storage[idx] = var

                if var is None:
                    old_inner_output_data[idx] = None
                elif isinstance(self.fn.maker.fgraph.outputs[idx], TensorVariable):
                    old_inner_output_data[idx] = var.data
                else:
                    raise RuntimeError(
                        "FIXME: old_inner_output_data[idx] = var.gpudata"
                    )

            # 4.6. Keep a reference to the variables (ndarrays,
            # etc) associated with mitmot inputs currently in the
            # input_storage to be able to compare them with the content of the
            # input_storage after the execution of the function. Also keep
            # pointers to their data to be able to detect cases where outputs
            # reused the allocated object but alter the memory region they
            # refer to.
            for idx in range(nb_mitmot_in):
                var = inner_input_storage[idx + info.n_seqs].storage[0]
                old_mitmot_input_storage[idx] = var

                if var is None:
                    old_mitmot_input_data[idx] = None
                else:
                    old_mitmot_input_data[idx] = var.data

            # 5.1 compute outputs
            t0_fn = time.perf_counter()

            try:
                vm()
            except Exception:
                if hasattr(vm, "position_of_error"):
                    # this is a new vm-provided function or c linker
                    # they need this because the exception manipulation
                    # done by raise_with_op is not implemented in C.
                    if hasattr(vm, "thunks"):
                        # For the CVM
                        raise_with_op(
                            self.fn.maker.fgraph,
                            vm.nodes[vm.position_of_error],
                            vm.thunks[vm.position_of_error],
                        )
                    else:
                        # For the c linker
                        # We don't have access from python to all the
                        # temps values So for now, we just don't print
                        # the extra shapes/strides info
                        raise_with_op(
                            self.fn.maker.fgraph, vm.nodes[vm.position_of_error]
                        )
                else:
                    # old-style linkers raise their own exceptions
                    raise

            dt_fn = time.perf_counter() - t0_fn
            if info.as_while:
                pdx = offset + info.n_shared_outs
                cond = inner_output_storage[pdx].storage[0] == 0

            t_fn += dt_fn
            offset_out = 0

            # 5.3 Copy over the values for mit_mot outputs
            mitmot_inp_grp_offset = 0
            mitmot_out_idx = 0
            for mitmot_grp_idx, taps in enumerate(info.mit_mot_in_slices):
                for out_slice in info.mit_mot_out_slices[mitmot_grp_idx]:
                    if self.mitmots_preallocated[mitmot_out_idx]:

                        mitmot_inp_idx = mitmot_inp_grp_offset + taps.index(out_slice)
                        inner_inp_idx = info.n_seqs + mitmot_inp_idx

                        # Verify whether the input points to the same data as
                        # it did before the execution of the inner function.
                        old_var = old_mitmot_input_storage[mitmot_inp_idx]
                        new_var = inner_input_storage[inner_inp_idx].storage[0]
                        if old_var is new_var:
                            old_data = old_mitmot_input_data[mitmot_inp_idx]
                            same_data = new_var.data == old_data
                        else:
                            same_data = False

                        # If the corresponding input storage still points to
                        # the same data, it has been modified inplace and
                        # nothing needs to be done. Otherwise, recover the
                        # and store it in `outs` as usual
                        if not same_data:
                            output_storage[mitmot_grp_idx][0][
                                out_slice + pos[mitmot_grp_idx]
                            ] = inner_input_storage[inner_inp_idx].storage[0]
                    else:
                        # This output tap has not been preallocated, recover
                        # its value as usual
                        output_storage[mitmot_grp_idx][0][
                            out_slice + pos[mitmot_grp_idx]
                        ] = inner_output_storage[offset_out].storage[0]

                    offset_out += 1
                    mitmot_out_idx += 1

                mitmot_inp_grp_offset += len(taps)

            # 5.4 Copy over the values for mit_sot/sit_sot outputs
            begin = info.n_mit_mot
            end = self.n_outs
            offset_out -= info.n_mit_mot

            for j in range(begin, end):

                # Copy the output value to `outs`, if necessary
                if store_steps[j] == 1 or self.vector_outs[j]:
                    output_storage[j][0][pos[j]] = inner_output_storage[
                        offset_out + j
                    ].storage[0]
                else:
                    # Check whether the initialization of the output storage
                    # map for this output has been reused.
                    old_var = old_inner_output_storage[offset_out + j]
                    new_var = inner_output_storage[offset_out + j].storage[0]
                    if old_var is new_var:
                        old_data = old_inner_output_data[offset_out + j]
                        if old_data is None:
                            output_reused = False
                        elif isinstance(
                            self.fn.maker.fgraph.outputs[offset_out + j], TensorVariable
                        ):
                            output_reused = new_var.data == old_data
                        else:
                            raise RuntimeError(
                                "FIXME: output_reused = new_var.gpudata == old_data"
                            )
                    else:
                        output_reused = False

                    if not output_reused:
                        try:
                            output_storage[j][0][pos[j]] = inner_output_storage[
                                offset_out + j
                            ].storage[0]
                        except ValueError as e:
                            if i == 0:
                                # First iteration, so don't change the
                                # error message as it can't be the
                                # case we write about.
                                raise
                            ne = ValueError(
                                "An output of the Scan has changed shape. "
                                "This may be caused by a push-out optimization."
                                " Try adding 'optimizer_excluding=scan_pushout'"
                                " to your Aesara flags."
                            )
                            raise ne from e

            # 5.5 Copy over the values for nit_sot outputs
            begin = end
            end += info.n_nit_sot
            for j in range(begin, end):

                if i == 0:
                    jout = j + offset_out
                    shape = (store_steps[j],) + inner_output_storage[jout].storage[
                        0
                    ].shape
                    dtype = inner_output_storage[jout].storage[0].dtype
                    if (
                        output_storage[j][0] is None
                        or output_storage[j][0].shape[0] < store_steps[j]
                        or output_storage[j][0].shape[1:] != shape[1:]
                        or output_storage[j][0].dtype != dtype
                    ):
                        output_storage[j][0] = np.empty(
                            shape, dtype=node.outputs[j].type.dtype
                        )
                    elif output_storage[j][0].shape[0] != store_steps[j]:
                        output_storage[j][0] = output_storage[j][0][: store_steps[j]]
                    output_storage[j][0][pos[j]] = inner_output_storage[jout].storage[0]
                elif store_steps[j] == 1 or self.vector_outs[j]:
                    output_storage[j][0][pos[j]] = inner_output_storage[
                        j + offset_out
                    ].storage[0]
                else:
                    # Check whether the initialization of the output storage map
                    # for this output has been reused.
                    old_var = old_inner_output_storage[offset_out + j]
                    old_data = old_inner_output_data[offset_out + j]
                    new_var = inner_output_storage[offset_out + j].storage[0]
                    if old_var is new_var:
                        if old_data is None:
                            output_reused = False
                        elif isinstance(
                            self.fn.maker.fgraph.outputs[offset_out + j], TensorVariable
                        ):
                            output_reused = new_var.data == old_data
                        else:
                            raise RuntimeError(
                                "FIXME: output_reused = new_var.gpudata == old_data"
                            )
                    else:
                        output_reused = False

                    if not output_reused:
                        output_storage[j][0][pos[j]] = inner_output_storage[
                            j + offset_out
                        ].storage[0]

            # 5.6 Copy over the values for outputs corresponding to shared
            # variables
            begin = end
            end += info.n_shared_outs
            for j in range(begin, end):
                jout = j + offset_out
                output_storage[j][0] = inner_output_storage[jout].storage[0]

            pos = [(idx + 1) % store for idx, store in zip(pos, store_steps)]
            i = i + 1

        # 6. Check if you need to re-order output buffers
        begin = info.n_mit_mot
        end = self.n_outs + info.n_nit_sot
        for idx in range(begin, end):
            if store_steps[idx] < i - self.mintaps[idx] and pos[idx] < store_steps[idx]:

                pdx = pos[idx]
                if pdx >= store_steps[idx] // 2:
                    # It seems inefficient to copy the bigger part of the
                    # array over, and back, but it is the only way that
                    # there is no overlap in the areas of out[idx][0] that
                    # are read and written.
                    # This way, there will be no information overwritten
                    # before it is read (as it used to happen).
                    shape = (pdx,) + output_storage[idx][0].shape[1:]
                    tmp = np.empty(shape, dtype=node.outputs[idx].type.dtype)
                    tmp[:] = output_storage[idx][0][:pdx]
                    output_storage[idx][0][: store_steps[idx] - pdx] = output_storage[
                        idx
                    ][0][pdx:]
                    output_storage[idx][0][store_steps[idx] - pdx :] = tmp
                    del tmp
                else:
                    shape = (store_steps[idx] - pdx,) + output_storage[idx][0].shape[1:]
                    tmp = np.empty(shape, dtype=node.outputs[idx].type.dtype)
                    tmp[:] = output_storage[idx][0][pdx:]
                    output_storage[idx][0][store_steps[idx] - pdx :] = output_storage[
                        idx
                    ][0][:pdx]
                    output_storage[idx][0][: store_steps[idx] - pdx] = tmp
                    del tmp
            # This would normally happen only when doing truncated
            # backpropagation through time. In such a scenario Scan is
            # expected to return 0 for all entries for which the gradient is
            # not actually computed
            elif store_steps[idx] > i - self.mintaps[idx]:
                output_storage[idx][0][i - self.mintaps[idx] :] = 0
                # This is a fix for a bug introduced by while. If you say
                # you want to loop up to a condition, you expect the output
                # to have that length ( and not the maximal length possible)
                #
                # Without this the behaviour of a scan op is not consistent
                # if optimization gets applied compared to when optimization
                # do not get applied
                if i < n_steps:
                    # The reason I don't use out[idx][0][:i] is because for
                    # certain outputs (those with multiple taps),
                    # outs[idx][0] has more than n_steps entries, with the
                    # initial state  at the beginning. When indexing in it I
                    # usually have to do something like
                    # outs[idx][0][i+offset]. To do something similar here,
                    # I would have first to compute the maximal tap for
                    # every output and then do outs[0][:i+maximal_tap],
                    # which implies I think more computations then this
                    # little trick that I used
                    output_storage[idx][0] = output_storage[idx][0][: -(n_steps - i)]

        # We never reuse the input or output storage of the
        # inner function so we clear it.
        for i_s in inner_input_storage:
            i_s.storage[0] = None
        for o_s in inner_output_storage:
            o_s.storage[0] = None

        t_call = time.perf_counter() - t0_call
        # NOTE: make this match what's in function.types.Function
        # and this little string helps us to find this spot:
        # "PROFILE_CODE"

        if hasattr(self.fn.maker, "profile") and self.fn.maker.profile:
            profile = self.fn.maker.profile
            profile.callcount += 1
            profile.nbsteps += n_steps
            profile.call_time += t_call
            profile.vm_call_time += t_fn
            if hasattr(self.fn.vm, "update_profile"):
                self.fn.vm.update_profile(profile)

        self.t_call = t_call
        self.t_fn = t_fn

    def infer_shape(self, fgraph, node, input_shapes):
        # input_shapes correspond to the shapes of node.inputs
        for inp, inp_shp in zip(node.inputs, input_shapes):
            assert inp_shp is None or len(inp_shp) == inp.type.ndim

        # Here we build 2 variables;
        # - A list `inner_ins_shapes`, such that inner_ins_shapes[i] is the
        #   shape of self.inner_inputs[i]
        # - A dictionary `out_equivalent` containing, for every inner input,
        #   an equivalent variable computed from the outer inputs.
        #   NOTE : For non-sequences, this equivalence is trivial. For
        #   sequences and recurrent states, there is no direct equivalence
        #   between outer and inner inputs. However, because every iteration
        #   of the Scan needs to give the same output shapes, we can give an
        #   equivalence between these inner inputs and the subelements of the
        #   corresponding outer inputs that the Scan would use as input for
        #   any given iteration. For simplicity, we use iteration 0.
        inner_ins_shapes = []
        out_equivalent = OrderedDict()

        # The two following blocks are commented as it cause in some
        # cases extra scans in the graph. See gh-XXX for the
        # investigation.

        info = self.info
        # We skip the first outer input as it is the total or current number
        # of iterations.
        # sequences
        seqs_shape = [x[1:] for x in input_shapes[1 : 1 + info.n_seqs]]
        # We disable extra infer_shape for now. See gh-3765.
        # if extra_infer_shape:
        #     inner_seqs = self.inner_inputs[: info.n_seqs]
        #     outer_seqs = node.inputs[1 : 1 + info.n_seqs]
        #     for in_s, out_s in zip(inner_seqs, outer_seqs):
        #         out_equivalent[in_s] = out_s[0]
        #
        #     # mit_mot, mit_sot, sit_sot
        #     outer_inp_idx = 1 + info.n_seqs
        #     inner_inp_idx = info.n_seqs
        # else:
        #     outer_inp_idx = 0
        outer_inp_idx = 0

        n_outs = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
        outs_shape = []
        for idx, taps in enumerate(
            chain(
                info.mit_mot_in_slices, info.mit_sot_in_slices, info.sit_sot_in_slices
            )
        ):
            for k in taps:
                outs_shape += [input_shapes[idx + info.n_seqs + 1][1:]]
                # if extra_infer_shape:
                #     mintap = abs(min(taps))
                #     corresponding_tap = node.inputs[outer_inp_idx][mintap + k]
                #     out_equivalent[self.inner_inputs[inner_inp_idx]] = corresponding_tap
            outer_inp_idx += 1

        # shared_outs
        offset = 1 + info.n_seqs + n_outs
        for idx in range(info.n_shared_outs):
            outs_shape += [input_shapes[idx + offset]]

        # non_sequences
        offset += info.n_nit_sot + info.n_shared_outs
        inner_ins_shapes = seqs_shape + outs_shape + input_shapes[offset:]
        assert len(inner_ins_shapes) == len(self.inner_inputs)

        # Non-sequences have a direct equivalent from self.inner_inputs in
        # node.inputs
        inner_non_sequences = self.inner_inputs[len(seqs_shape) + len(outs_shape) :]
        for in_ns, out_ns in zip(inner_non_sequences, node.inputs[offset:]):
            out_equivalent[in_ns] = out_ns

        if info.as_while:
            self_outs = self.inner_outputs[:-1]
        else:
            self_outs = self.inner_outputs
        outs_shape = infer_shape(
            outs=self_outs, inputs=self.inner_inputs, input_shapes=inner_ins_shapes
        )
        # Will be used to check if outs_shape can be expressed without using
        # variables in self.inner_inputs.
        # The shapes of node.inputs are valid.
        validator = Validator(
            valid=input_shapes,
            invalid=self.inner_inputs,
            valid_equivalent=out_equivalent,
        )

        offset = 1 + info.n_seqs
        scan_outs = [x for x in input_shapes[offset : offset + n_outs]]
        offset += n_outs
        outs_shape_n = info.n_mit_mot_outs + info.n_mit_sot + info.n_sit_sot
        for x in range(info.n_nit_sot):
            out_shape_x = outs_shape[outs_shape_n + x]
            if out_shape_x is None:
                # This output is not a tensor, and has no shape
                scan_outs.append(None)
            else:
                # We need to make sure that we can compute the shapes from
                # node.inputs, and constants, without using the variables
                # in the inner function.
                r = node.outputs[n_outs + x]
                assert r.ndim == 1 + len(out_shape_x)
                shp = [node.inputs[offset + info.n_shared_outs + x]]
                for i, shp_i in zip(range(1, r.ndim), out_shape_x):
                    # Validate shp_i. v_shape_i is either None (if invalid),
                    # or a (variable, Boolean) tuple. The Boolean indicates
                    # whether variable is shp_i (if True), or an valid
                    # equivalent (if False). Here, we only need the variable.
                    v_shp_i = validator.check(shp_i)
                    if v_shp_i is None:
                        if r.type.shape[i] is not None:
                            shp.append(r.type.shape[i])
                        else:
                            shp.append(Shape_i(i)(r))
                    else:
                        # It can (or at least, an equivalent variable can)
                        shp.append(v_shp_i[0])
                scan_outs.append(tuple(shp))

        scan_outs += [x for x in input_shapes[offset : offset + info.n_shared_outs]]
        # if we are dealing with a repeat-until, then we do not know the
        # leading dimension so we replace it for every entry with Shape_i
        if info.as_while:
            scan_outs_init = scan_outs
            scan_outs = []
            for o, x in zip(node.outputs, scan_outs_init):
                if x is None:
                    scan_outs.append(None)
                else:
                    scan_outs.append((Shape_i(0)(o),) + x[1:])
        return scan_outs

    def connection_pattern(self, node):

        # We cache the result of this function because, with a previous
        # implementation that repeatedly called grad, there were cases
        # where calls to aesara.grad() took as much as 4h for functions
        # containing many nested scans.
        if hasattr(node.tag, "connection_pattern"):
            return node.tag.connection_pattern

        # Obtain the connection pattern of the inner function.
        inner_connect_pattern = io_connection_pattern(
            self.inner_inputs, self.inner_outputs
        )

        # Initially assume no outer input is connected to any outer output
        connection_pattern = [[False for output in node.outputs] for x in node.inputs]

        # For every possible pair of outer input and outer output, iterate
        # over every possible pairing of their corresponding inner inputs
        # and inner outputs and, if one such pair of inner variables is
        # connected than the pair of outer variables is connected.
        var_mappings = self.get_oinp_iinp_iout_oout_mappings()

        for outer_oidx in range(len(node.outputs)):
            inner_oidxs = var_mappings["inner_out_from_outer_out"][outer_oidx]

            for outer_iidx in range(len(node.inputs)):
                inner_iidxs = var_mappings["inner_inp_from_outer_inp"][outer_iidx]

                for inner_oidx in inner_oidxs:
                    for inner_iidx in inner_iidxs:

                        if inner_connect_pattern[inner_iidx][inner_oidx]:
                            connection_pattern[outer_iidx][outer_oidx] = True
                            break

                    if connection_pattern[outer_iidx][outer_oidx]:
                        break

        # Applying Floyd-Warshall to find all paths connecting inputs to
        # outputs. Note that if `x` is an input to `y_t` and `y_tm1` is an
        # input to `z_t` then `x` is an input to `z_t`.

        n_outs = len(node.outputs)

        for steps in range(n_outs):
            for iidx in range(n_outs):
                for jidx in range(n_outs):

                    # Get the idx of the outer input corresponding to that
                    # outer output
                    j_inp_idx = var_mappings["outer_inp_from_outer_out"][jidx]

                    if j_inp_idx != -1:
                        if connection_pattern[j_inp_idx][iidx] is True:
                            for k in range(len(connection_pattern)):
                                if connection_pattern[k][jidx]:
                                    connection_pattern[k][iidx] = True

        node.tag.connection_pattern = connection_pattern
        return connection_pattern

    def L_op(self, inputs, outs, dC_douts):
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        # `grad_step` equals the number of steps the original scan node has
        # done (if the original scan is a while loop than this number is the
        # length of the output sequence)
        # We do not know what kind of outputs the original scan has, so we
        # try first to see if it has a nit_sot output, then a sit_sot and
        # then a mit_sot
        info = self.info
        if info.n_nit_sot > 0:
            grad_steps = self.outer_nitsot_outs(outs)[0].shape[0]
        elif info.n_sit_sot > 0:
            grad_steps = self.outer_sitsot_outs(outs)[0].shape[0] - 1
        elif info.n_mit_sot > 0:
            grad_steps = (
                self.outer_mitsot_outs(outs)[0].shape[0] + self.mintaps[info.n_mit_mot]
            )
        else:
            grad_steps = inputs[0]
        if info.as_while:
            n_steps = outs[0].shape[0]

        # Restrict the number of grad steps according to
        # self.truncate_gradient
        if self.truncate_gradient != -1:
            grad_steps = minimum(grad_steps, self.truncate_gradient)

        self_inputs = self.inner_inputs
        self_outputs = self.inner_outputs
        # differentiable inputs
        diff_inputs = (
            self.inner_seqs(self_inputs)
            + self.inner_mitmot(self_inputs)
            + self.inner_mitsot(self_inputs)
            + self.inner_sitsot(self_inputs)
            + self.inner_non_seqs(self_inputs)
        )
        diff_outputs = (
            self.inner_mitmot_outs(self_outputs)
            + self.inner_mitsot_outs(self_outputs)
            + self.inner_sitsot_outs(self_outputs)
            + self.inner_nitsot_outs(self_outputs)
        )
        scan_node = outs[0].owner
        connection_pattern = self.connection_pattern(scan_node)

        def get_inp_idx(iidx):
            if iidx < info.n_seqs:
                return 1 + iidx
            oidx = 1 + info.n_seqs
            iidx = iidx - info.n_seqs
            for taps in info.mit_mot_in_slices:
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)
            for taps in info.mit_sot_in_slices:
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)

            if iidx < info.n_sit_sot:
                return oidx + iidx
            else:
                return oidx + iidx + info.n_nit_sot

        def get_out_idx(iidx):
            oidx = 0
            for taps in info.mit_mot_out_slices[: info.n_mit_mot]:
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)
            return oidx + iidx

        def compute_all_gradients(known_grads):
            y_s = known_grads.keys()
            g_y_s = known_grads.values()

            for g_y in g_y_s:
                if str(g_y.dtype) in integer_dtypes:
                    raise TypeError(
                        "Gradients may never be integers but g_y "
                        f"has type {g_y.type}"
                    )

            out_indices = [get_out_idx(self_outputs.index(y)) for y in y_s]

            connected_inputs = [
                i
                for i in range(len(scan_node.inputs))
                if any(connection_pattern[i][odx] for odx in out_indices)
            ]

            wrt = [
                x
                for x in graph_inputs(y_s)
                if (x in diff_inputs)
                and get_inp_idx(self_inputs.index(x)) in connected_inputs
            ]
            gmp = OrderedDict()

            # Required in case there is a pair of variables X and Y, with X
            # used to compute Y, for both of which there is an external
            # gradient signal. Without this, the total gradient signal on X
            # will be the external gradient  signalknown_grads[X]. With this,
            # it will be the sum of the external gradient signal and the
            # gradient obtained by propagating Y's external gradient signal
            # to X.
            known_grads = OrderedDict([(k.copy(), v) for (k, v) in known_grads.items()])

            grads = grad(
                cost=None,
                known_grads=known_grads,
                wrt=wrt,
                consider_constant=wrt,
                disconnected_inputs="ignore",
                return_disconnected="None",
                null_gradients="return",
            )

            for i in range(len(wrt)):
                gmp[wrt[i]] = grads[i]

            rval = [gmp.get(p, None) for p in diff_inputs]
            return rval

        var_mappings = self.get_oinp_iinp_iout_oout_mappings()
        dC_dinps_t = [None for inp in diff_inputs]
        disconnected_dC_dinps_t = [True for inp in diff_inputs]
        dC_dXts = []
        Xts = []
        for idx, Xt in enumerate(diff_outputs):

            # We are looking for x[t-1] for a given x[t]
            if idx >= info.n_mit_mot_outs:
                Xt_placeholder = safe_new(Xt)
                Xts.append(Xt_placeholder)

            # Different processing based on whether Xt is a nitsot output
            # or not. NOTE : This cannot be done by using
            # "if Xt not in self.inner_nitsot_outs(self_outputs)" because
            # the exact same variable can be used as multiple outputs.
            idx_nitsot_start = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
            idx_nitsot_end = idx_nitsot_start + info.n_nit_sot
            if idx < idx_nitsot_start or idx >= idx_nitsot_end:
                # What we do here is loop through dC_douts and collect all
                # those that are connected to the specific one and do an
                # upcast on all of their dtypes to get the dtype for this
                # specific output. Deciding if the gradient with this
                # specific previous step is defined or not is done somewhere
                # else.
                dtypes = []
                states = (
                    self.inner_mitmot(self_inputs)
                    + self.inner_mitsot(self_inputs)
                    + self.inner_sitsot(self_inputs)
                )

                for pos, inp in enumerate(states):
                    if inp in graph_inputs([Xt]):
                        # Get the index of the outer output that to which
                        # the state variable 'inp' corresponds.
                        outer_oidx = var_mappings["outer_out_from_inner_inp"][
                            info.n_seqs + pos
                        ]

                        if not isinstance(dC_douts[outer_oidx].type, DisconnectedType):
                            dtypes.append(dC_douts[outer_oidx].dtype)
                if dtypes:
                    new_dtype = aesara.scalar.upcast(*dtypes)
                else:
                    new_dtype = config.floatX
                dC_dXt = safe_new(Xt, dtype=new_dtype)
            else:
                if isinstance(dC_douts[idx].type, DisconnectedType):
                    continue
                dC_dXt = safe_new(dC_douts[idx][0])
            dC_dXts.append(dC_dXt)

        known_grads = OrderedDict()
        dc_dxts_idx = 0
        for i in range(len(diff_outputs)):
            if i < idx_nitsot_start or i >= idx_nitsot_end:
                if diff_outputs[i] in known_grads:
                    known_grads[diff_outputs[i]] += dC_dXts[dc_dxts_idx]
                else:
                    known_grads[diff_outputs[i]] = dC_dXts[dc_dxts_idx]
                dc_dxts_idx += 1
            else:
                if isinstance(dC_douts[i].type, DisconnectedType):
                    continue
                else:
                    if diff_outputs[i] in known_grads:
                        known_grads[diff_outputs[i]] += dC_dXts[dc_dxts_idx]
                    else:
                        known_grads[diff_outputs[i]] = dC_dXts[dc_dxts_idx]
                    dc_dxts_idx += 1
        dC_dinps_t = compute_all_gradients(known_grads)

        # mask inputs that get no gradients
        for dx in range(len(dC_dinps_t)):
            if not dC_dinps_t[dx]:
                dC_dinps_t[dx] = at.zeros_like(diff_inputs[dx])
            else:
                disconnected_dC_dinps_t[dx] = False
                for Xt, Xt_placeholder in zip(diff_outputs[info.n_mit_mot_outs :], Xts):
                    tmp = forced_replace(dC_dinps_t[dx], Xt, Xt_placeholder)
                    dC_dinps_t[dx] = tmp

        # construct dX_dtm1
        dC_dXtm1s = []
        for pos, x in enumerate(dC_dinps_t[info.n_seqs :]):

            # Get the index of the first inner input corresponding to the
            # pos-ieth inner input state
            idxs = var_mappings["inner_out_from_inner_inp"][info.n_seqs + pos]

            # Check if the pos-th input is associated with one of the
            # recurrent states
            x_is_state = pos < sum(
                len(t)
                for t in chain(
                    info.mit_mot_in_slices,
                    info.mit_sot_in_slices,
                    info.sit_sot_in_slices,
                )
            )

            if x_is_state and len(idxs) > 0:
                opos = idxs[0]
                dC_dXtm1s.append(safe_new(dC_dXts[opos]))
                if hasattr(x, "dtype") and x.dtype != dC_dXts[opos].dtype:
                    dC_dinps_t[pos + info.n_seqs] = x.astype(dC_dXts[opos].dtype)
            else:
                dC_dXtm1s.append(safe_new(x))

        for dx, dC_dXtm1 in enumerate(dC_dXtm1s):
            if isinstance(dC_dinps_t[dx + info.n_seqs].type, NullType):
                # The accumulated gradient is undefined
                pass
            elif isinstance(dC_dXtm1.type, NullType):
                # The new gradient is undefined, this makes the accumulated
                # gradient undefined as weell
                dC_dinps_t[dx + info.n_seqs] = dC_dXtm1
            else:
                dC_dinps_t[dx + info.n_seqs] += dC_dXtm1

        if info.as_while:
            # equivalent to x[:n_steps][::-1]
            outer_inp_seqs = [x[n_steps - 1 :: -1] for x in inputs[1 : 1 + info.n_seqs]]
        else:
            outer_inp_seqs = [x[::-1] for x in inputs[1 : 1 + info.n_seqs]]
        for idx, taps in enumerate(
            chain(info.mit_mot_in_slices, info.mit_sot_in_slices)
        ):
            mintap = min(taps)
            if idx < info.n_mit_mot:
                outmaxtap = np.max(info.mit_mot_out_slices[: info.n_mit_mot][idx])
            else:
                outmaxtap = 0
            seq = outs[idx]
            for k in taps:
                if outmaxtap - k != 0:
                    nw_seq = seq[k - mintap : -(outmaxtap - k)][::-1]
                else:
                    nw_seq = seq[k - mintap :][::-1]
                outer_inp_seqs.append(nw_seq)
        outer_inp_seqs += [x[:-1][::-1] for x in self.outer_sitsot_outs(outs)]
        for x in self.outer_nitsot_outs(dC_douts):
            if not isinstance(x.type, DisconnectedType):
                if info.as_while:
                    # equivalent to x[:n_steps][::-1]
                    outer_inp_seqs.append(x[n_steps - 1 :: -1])
                else:
                    outer_inp_seqs.append(x[::-1])

        if hasattr(inputs[0].tag, "test_value"):
            # Here we tests that the new scan input sequence all have
            # the same shape[0]. This is a properties that the scan()
            # fct add and we want to keep it for all Scan op.  This is
            # used in T_Scan.test_grad_multiple_outs_taps to test
            # that.
            if info.as_while:
                n = n_steps.tag.test_value
            else:
                n = inputs[0].tag.test_value
            for taps, x in zip(info.mit_sot_in_slices, self.outer_mitsot_outs(outs)):
                mintap = np.min(taps)
                if hasattr(x[::-1][:mintap], "test_value"):
                    assert x[::-1][:mintap].tag.test_value.shape[0] == n
            for x in self.outer_sitsot_outs(outs):
                if hasattr(x[::-1][:-1].tag, "test_value"):
                    assert x[::-1][:-1].tag.test_value.shape[0] == n
            for x in self.outer_nitsot_outs(outs):
                if hasattr(x[::-1].tag, "test_value"):
                    if info.as_while:
                        assert x[n_steps - 1 :: -1].tag.test_value.shape[0] == n
                    else:
                        assert x[::-1].tag.test_value.shape[0] == n
        outer_inp_seqs += [
            x[::-1][: np.min(taps)]
            for taps, x in zip(info.mit_sot_in_slices, self.outer_mitsot_outs(outs))
        ]
        outer_inp_seqs += [x[::-1][:-1] for x in self.outer_sitsot_outs(outs)]
        outer_inp_seqs += [x[::-1] for x in self.outer_nitsot_outs(outs)]

        # Restrict the length of the outer sequences to the number of grad
        # steps
        outer_inp_seqs = [s_[:grad_steps] for s_ in outer_inp_seqs]

        inner_inp_seqs = self.inner_seqs(self_inputs)
        inner_inp_seqs += self.inner_mitmot(self_inputs)
        inner_inp_seqs += self.inner_mitsot(self_inputs)
        inner_inp_seqs += self.inner_sitsot(self_inputs)
        inner_inp_seqs += self.inner_nitsot_outs(dC_dXts)
        inner_inp_seqs += Xts
        # mitmot
        outer_inp_mitmot = []
        inner_inp_mitmot = []
        inner_out_mitmot = []
        mitmot_inp_taps = []
        mitmot_out_taps = []
        type_outs = []
        out_pos = 0
        ins_pos = info.n_seqs
        n_mitmot_outs = 0
        n_mitmot_inps = 0

        for idx, taps in enumerate(info.mit_mot_in_slices):
            if isinstance(dC_douts[idx].type, DisconnectedType):
                out = outs[idx]
                outer_inp_mitmot.append(at.zeros_like(out))
            else:
                outer_inp_mitmot.append(dC_douts[idx][::-1])
            mitmot_inp_taps.append([])
            mitmot_out_taps.append([])
            undefined_msg = None
            through_shared = False
            disconnected = True

            for mit_mot_out_slice in info.mit_mot_out_slices[idx]:
                inner_inp_mitmot.append(dC_dXts[out_pos])
                mitmot_inp_taps[idx].append(-mit_mot_out_slice)
                n_mitmot_inps += 1
                out_pos += 1

            for tap in taps:
                tap = -tap

                # Only create a new inner input if there is not already one
                # associated with this input tap
                if tap not in mitmot_inp_taps[idx]:
                    inner_inp_mitmot.append(dC_dXtm1s[ins_pos - info.n_seqs])

                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # We cannot use Null in the inner graph, so we
                    # use a zero tensor of the appropriate shape instead.
                    inner_out_mitmot.append(
                        at.zeros(diff_inputs[ins_pos].shape, dtype=config.floatX)
                    )
                    undefined_msg = dC_dinps_t[ins_pos].type.why_null
                else:
                    new_inner_out_mitmot = dC_dinps_t[ins_pos]

                    # If there is already an inner input associated with that
                    # input tap, make sure the computation of the new output
                    # uses it instead of the input it's currently using
                    if tap in mitmot_inp_taps[idx]:
                        to_replace = dC_dXtm1s[ins_pos - info.n_seqs]
                        replacement_idx = len(mitmot_inp_taps[idx]) - mitmot_inp_taps[
                            idx
                        ].index(tap)
                        replacement = inner_inp_mitmot[-replacement_idx]
                        new_inner_out_mitmot = clone_replace(
                            new_inner_out_mitmot, replace=[(to_replace, replacement)]
                        )

                    inner_out_mitmot.append(new_inner_out_mitmot)

                disconnected &= disconnected_dC_dinps_t[ins_pos]

                through_shared = any(
                    _sh in graph_inputs([dC_dinps_t[ins_pos]])
                    for _sh in self.inner_shared(self_inputs)
                )

                ins_pos += 1
                n_mitmot_outs += 1
                mitmot_out_taps[idx].append(tap)

                # Only add the tap as a new input tap if needed
                if tap not in mitmot_inp_taps[idx]:
                    n_mitmot_inps += 1
                    mitmot_inp_taps[idx].append(tap)

            if undefined_msg:
                type_outs.append(undefined_msg)
            elif through_shared:
                type_outs.append("through_shared")
            elif disconnected:
                type_outs.append("disconnected")
            else:
                type_outs.append("connected")

        offset = info.n_mit_mot
        for idx, taps in enumerate(info.mit_sot_in_slices):
            if isinstance(dC_douts[idx + offset].type, DisconnectedType):
                outer_inp_mitmot.append(outs[idx + offset].zeros_like())
            else:
                outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            mitmot_inp_taps.append([])
            mitmot_out_taps.append([])
            inner_inp_mitmot.append(dC_dXts[out_pos])
            out_pos += 1
            n_mitmot_inps += 1
            undefined_msg = None
            through_shared = False
            disconnected = True
            mitmot_inp_taps[idx + offset].append(0)
            for tap in taps:
                tap = -tap
                inner_inp_mitmot.append(dC_dXtm1s[ins_pos - info.n_seqs])

                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # We cannot use Null in the inner graph, so we
                    # use a zero tensor of the appropriate shape instead.
                    inner_out_mitmot.append(
                        at.zeros(diff_inputs[ins_pos].shape, dtype=config.floatX)
                    )
                    undefined_msg = dC_dinps_t[ins_pos].type.why_null
                else:
                    inner_out_mitmot.append(dC_dinps_t[ins_pos])

                mitmot_inp_taps[idx + offset].append(tap)
                mitmot_out_taps[idx].append(tap)

                disconnected &= disconnected_dC_dinps_t[ins_pos]

                through_shared = any(
                    _sh in graph_inputs([dC_dinps_t[ins_pos]])
                    for _sh in self.inner_shared(self_inputs)
                )

                n_mitmot_inps += 1
                ins_pos += 1
                n_mitmot_outs += 1

            if undefined_msg:
                type_outs.append(undefined_msg)
            elif through_shared:
                type_outs.append("through_shared")
            elif disconnected:
                type_outs.append("disconnected")
            else:
                type_outs.append("connected")

        offset += info.n_mit_sot
        for idx in range(info.n_sit_sot):
            mitmot_inp_taps.append([0, 1])
            mitmot_out_taps.append([1])
            through_shared = False
            if not isinstance(dC_douts[idx + offset].type, DisconnectedType):
                outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            else:
                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # Cannot use dC_dinps_t[ins_pos].dtype, so we use
                    # floatX instead, as it is a dummy value that will not
                    # be used anyway.
                    outer_inp_mitmot.append(
                        at.zeros(outs[idx + offset].shape, dtype=config.floatX)
                    )
                else:
                    outer_inp_mitmot.append(
                        at.zeros(
                            outs[idx + offset].shape, dtype=dC_dinps_t[ins_pos].dtype
                        )
                    )

            if isinstance(dC_dinps_t[ins_pos].type, NullType):
                # We cannot use Null in the inner graph, so we
                # use a zero tensor of the appropriate shape instead.
                inner_out_mitmot.append(
                    at.zeros(diff_inputs[ins_pos].shape, dtype=config.floatX)
                )
            else:
                inner_out_mitmot.append(dC_dinps_t[ins_pos])

            through_shared = any(
                _sh in graph_inputs([dC_dinps_t[ins_pos]])
                for _sh in self.inner_shared(self_inputs)
            )

            if isinstance(dC_dinps_t[ins_pos].type, NullType):
                type_outs.append(dC_dinps_t[ins_pos].type.why_null)
            elif through_shared:
                type_outs.append("through_shared")
            elif disconnected_dC_dinps_t[ins_pos]:
                type_outs.append("disconnected")
            else:
                type_outs.append("connected")

            inner_inp_mitmot += [
                dC_dXts[out_pos],
                dC_dXtm1s[ins_pos - info.n_seqs],
            ]
            n_mitmot_outs += 1
            out_pos += 1
            ins_pos += 1
            n_mitmot_inps += 2

        n_nit_sot = info.n_seqs
        inner_out_nitsot = dC_dinps_t[: info.n_seqs]
        inner_out_sitsot = dC_dinps_t[ins_pos:]
        for _p, vl in enumerate(inner_out_sitsot):
            through_shared = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in graph_inputs([vl]):
                    through_shared = True
            if isinstance(vl.type, NullType):
                type_outs.append(vl.type.why_null)
                # Replace the inner output with a zero tensor of
                # the right shape
                inner_out_sitsot[_p] = at.zeros(
                    diff_inputs[ins_pos + _p].shape, dtype=config.floatX
                )
            elif through_shared:
                type_outs.append("through_shared")
            elif disconnected_dC_dinps_t[_p + ins_pos]:
                type_outs.append("disconnected")
            else:
                type_outs.append("connected")

        for _p, vl in enumerate(inner_out_nitsot):
            through_shared = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in graph_inputs([vl]):
                    through_shared = True
            if isinstance(vl.type, NullType):
                type_outs.append(vl.type.why_null)
                # Replace the inner output with a zero tensor of
                # the right shape
                inner_out_nitsot[_p] = at.zeros(
                    diff_inputs[_p].shape, dtype=config.floatX
                )

            if through_shared:
                type_outs.append("through_shared")
            elif disconnected_dC_dinps_t[_p]:
                type_outs.append("disconnected")
            else:
                type_outs.append("connected")

        inner_inp_sitsot = dC_dXtm1s[ins_pos - info.n_seqs :]
        outer_inp_sitsot = []
        for _idx, y in enumerate(inner_inp_sitsot):
            x = self.outer_non_seqs(inputs)[_idx]
            if isinstance(y.type, NullType):
                # Cannot use dC_dXtm1s.dtype, so we use floatX instead.
                outer_inp_sitsot.append(
                    at.zeros(
                        [grad_steps + 1] + [x.shape[i] for i in range(x.ndim)],
                        dtype=config.floatX,
                    )
                )
                # replace y by a zero tensor of the right shape
                inner_inp_sitsot[_idx] = at.zeros(
                    diff_inputs[ins_pos + _idx].shape, dtype=config.floatX
                )

            else:
                outer_inp_sitsot.append(
                    at.zeros(
                        [grad_steps + 1] + [x.shape[i] for i in range(x.ndim)],
                        dtype=y.dtype,
                    )
                )

        n_sitsot_outs = len(outer_inp_sitsot)

        outer_inputs = (
            [grad_steps]
            + outer_inp_seqs
            + outer_inp_mitmot
            + outer_inp_sitsot
            + [n_steps if info.as_while else inputs[0] for _ in range(n_nit_sot)]
            + self.outer_shared(inputs)
            + self.outer_non_seqs(inputs)
        )

        inner_gfn_ins = (
            inner_inp_seqs
            + inner_inp_mitmot
            + inner_inp_sitsot
            + self.inner_shared(self_inputs)
            + self.inner_non_seqs(self_inputs)
        )
        inner_gfn_outs = inner_out_mitmot + inner_out_sitsot + inner_out_nitsot

        out_info = ScanInfo(
            n_seqs=len(outer_inp_seqs),
            mit_mot_in_slices=tuple(tuple(v) for v in mitmot_inp_taps),
            mit_mot_out_slices=tuple(tuple(v) for v in mitmot_out_taps),
            mit_sot_in_slices=(),
            sit_sot_in_slices=tuple((-1,) for k in range(n_sitsot_outs)),
            n_nit_sot=n_nit_sot,
            n_shared_outs=0,
            n_non_seqs=len(self.outer_shared(inputs))
            + len(self.outer_non_seqs(inputs)),
            as_while=False,
        )

        local_op = Scan(
            inner_gfn_ins,
            inner_gfn_outs,
            out_info,
            mode=self.mode,
            truncate_gradient=self.truncate_gradient,
            profile=self.profile,
            name=f"grad_of_{self.name}" if self.name else None,
            allow_gc=self.allow_gc,
        )
        outputs = local_op(*outer_inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        # Re-order the gradients correctly
        gradients = [DisconnectedType()()]

        offset = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot + n_sitsot_outs
        for p, (x, t) in enumerate(
            zip(
                outputs[offset : offset + info.n_seqs],
                type_outs[offset : offset + info.n_seqs],
            )
        ):
            if t == "connected":
                # If the forward scan is in as_while mode, we need to pad
                # the gradients, so that they match the size of the input
                # sequences.
                if info.as_while:
                    n_zeros = inputs[0] - n_steps
                    shp = (n_zeros,)
                    if x.ndim > 1:
                        shp = shp + tuple(x.shape[i] for i in range(1, x.ndim))
                    z = at.zeros(shp, dtype=x.dtype)
                    x = at.concatenate([x[::-1], z], axis=0)
                    gradients.append(x)
                else:
                    gradients.append(x[::-1])
            elif t == "disconnected":
                gradients.append(DisconnectedType()())
            elif t == "through_shared":
                gradients.append(
                    grad_undefined(
                        self, p + 1, inputs[p + 1], "Depends on a shared variable"
                    )
                )
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        end = info.n_mit_mot + info.n_mit_sot + info.n_sit_sot
        for p, (x, t) in enumerate(zip(outputs[:end], type_outs[:end])):
            if t == "connected":
                # If the forward scan is in as_while mode, we need to pad
                # the gradients, so that they match the size of the input
                # sequences.
                if info.as_while:
                    n_zeros = inputs[0] - grad_steps
                    shp = (n_zeros,)
                    if x.ndim > 1:
                        shp = shp + tuple(x.shape[i] for i in range(1, x.ndim))
                    z = at.zeros(shp, dtype=x.dtype)
                    x = at.concatenate([x[::-1], z], axis=0)
                    gradients.append(x)
                else:
                    gradients.append(x[::-1])
            elif t == "disconnected":
                gradients.append(DisconnectedType()())
            elif t == "through_shared":
                gradients.append(
                    grad_undefined(
                        self,
                        p + 1 + info.n_seqs,
                        inputs[p + 1 + info.n_seqs],
                        "Depends on a shared variable",
                    )
                )
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        start = len(gradients)
        node = outs[0].owner
        for idx in range(info.n_shared_outs):
            disconnected = True
            connected_flags = self.connection_pattern(node)[idx + start]
            for dC_dout, connected in zip(dC_douts, connected_flags):
                if not isinstance(dC_dout.type, DisconnectedType) and connected:
                    disconnected = False
            if disconnected:
                gradients.append(DisconnectedType()())
            else:
                gradients.append(
                    grad_undefined(
                        self, idx, inputs[idx], "Shared Variable with update"
                    )
                )

        start = len(gradients)
        gradients += [DisconnectedType()() for _ in range(info.n_nit_sot)]
        begin = end

        end = begin + n_sitsot_outs
        for p, (x, t) in enumerate(zip(outputs[begin:end], type_outs[begin:end])):
            if t == "connected":
                gradients.append(x[-1])
            elif t == "disconnected":
                gradients.append(DisconnectedType()())
            elif t == "through_shared":
                gradients.append(
                    grad_undefined(
                        self,
                        p + begin + 1,
                        inputs[p + begin + 1],
                        "Depends on a shared variable",
                    )
                )
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        # Mask disconnected gradients
        # Ideally we would want to assert that the gradients we are
        # replacing do indeed evaluate to 0, though that is not practical
        # from a computational point of view
        # The gradients of scan are computed replacing Disconnected with 0,
        # because through the recurrence they can become nonzero
        for idx in range(len(gradients)):
            disconnected = True
            for kdx in range(len(node.outputs)):
                if connection_pattern[idx][kdx] and not isinstance(
                    dC_douts[kdx].type, DisconnectedType
                ):
                    disconnected = False
            if disconnected:
                gradients[idx] = DisconnectedType()()
        return gradients

    def R_op(self, inputs, eval_points):
        # Step 0. Prepare some shortcut variable
        info = self.info
        self_inputs = self.inner_inputs
        rop_of_inputs = (
            self_inputs[: info.n_seqs + self.n_outs]
            + self_inputs[info.n_seqs + self.n_outs + info.n_shared_outs :]
        )
        self_outputs = self.inner_outputs

        # Step 1. Compute the R_op of the inner function
        inner_eval_points = [safe_new(x, "_evalpoint") for x in rop_of_inputs]
        if info.as_while:
            rop_self_outputs = self_outputs[:-1]
        else:
            rop_self_outputs = self_outputs
        if info.n_shared_outs > 0:
            rop_self_outputs = rop_self_outputs[: -info.n_shared_outs]
        rop_outs = Rop(rop_self_outputs, rop_of_inputs, inner_eval_points)
        if not isinstance(rop_outs, (list, tuple)):
            rop_outs = [rop_outs]
        # Step 2. Figure out what corresponds to what in the scan

        # When doing the R-op of scan, you end up having double of each type of
        # input, because for each sequence you need also its eval point, for
        # each mit_mot, mit_sot, sit_sot or other type of inputs the same.
        # Interestingly enough, all these types of eval points behave the same
        # way as the input to which they correspond
        # The only exception is the eval point for the number of sequences, and
        # evan point for the number of nit_sot which I think should just be
        # ignored (?)

        new_mit_mot_in_slices = info.mit_mot_in_slices * 2
        new_mit_sot_in_slices = info.mit_sot_in_slices * 2
        new_sit_sot_in_slices = info.sit_sot_in_slices * 2

        # Sequences ...
        b = 1
        ib = 0
        e = 1 + info.n_seqs
        ie = info.n_seqs
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_seqs = inputs[b:e] + clean_eval_points
        inner_seqs = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_MOT sequences ...
        b = e
        e = e + info.n_mit_mot
        ib = ie
        ie = ie + int(sum(len(x) for x in info.mit_mot_in_slices))
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_mit_mot = inputs[b:e] + clean_eval_points
        inner_mit_mot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_SOT sequences ...
        b = e
        e = e + info.n_mit_sot
        ib = ie
        ie = ie + int(sum(len(x) for x in info.mit_sot_in_slices))
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_mit_sot = inputs[b:e] + eval_points[b:e]
        inner_mit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # SIT_SOT sequences ...
        b = e
        e = e + info.n_sit_sot
        ib = ie
        ie = ie + info.n_sit_sot
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_sit_sot = inputs[b:e] + clean_eval_points
        inner_sit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # Shared outs ...
        b = e
        e = e + info.n_shared_outs
        ib = ie
        ie = ie + info.n_shared_outs
        scan_shared = inputs[b:e]
        inner_shared = self_inputs[ib:ie]

        # NIT_SOT sequences
        b = e
        e = e + info.n_nit_sot
        scan_nit_sot = inputs[b:e] * 2

        # All other arguments
        clean_eval_points = []
        for inp, evp in zip(inputs[e:], eval_points[e:]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())
        scan_other = inputs[e:] + clean_eval_points
        # inner_eval_points do not have entries for shared variables
        inner_other = self_inputs[ie:] + inner_eval_points[ib:]

        # Outputs
        n_mit_mot_outs = int(sum(len(x) for x in info.mit_mot_out_slices))

        b = 0
        e = n_mit_mot_outs
        inner_out_mit_mot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + info.n_mit_sot
        inner_out_mit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + info.n_sit_sot
        inner_out_sit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + info.n_nit_sot
        inner_out_nit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + info.n_shared_outs
        inner_out_shared = self_outputs[b:e]

        inner_ins = (
            inner_seqs
            + inner_mit_mot
            + inner_mit_sot
            + inner_sit_sot
            + inner_shared
            + inner_other
        )
        inner_outs = (
            inner_out_mit_mot
            + inner_out_mit_sot
            + inner_out_sit_sot
            + inner_out_nit_sot
            + inner_out_shared
        )

        if info.as_while:
            inner_outs += [self_outputs[-1]]
        scan_inputs = (
            [inputs[0]]
            + scan_seqs
            + scan_mit_mot
            + scan_mit_sot
            + scan_sit_sot
            + scan_shared
            + scan_nit_sot
            + scan_other
        )

        out_info = ScanInfo(
            n_seqs=info.n_seqs * 2,
            mit_mot_in_slices=new_mit_mot_in_slices,
            mit_mot_out_slices=tuple(tuple(v) for v in info.mit_mot_out_slices) * 2,
            mit_sot_in_slices=new_mit_sot_in_slices,
            sit_sot_in_slices=new_sit_sot_in_slices,
            n_nit_sot=info.n_nit_sot * 2,
            n_shared_outs=info.n_shared_outs,
            n_non_seqs=len(inner_other),
            as_while=info.as_while,
        )

        local_op = Scan(
            inner_ins,
            inner_outs,
            out_info,
            mode=self.mode,
            profile=self.profile,
            truncate_gradient=self.truncate_gradient,
            name=f"rop_of_{self.name}" if self.name else None,
            allow_gc=self.allow_gc,
        )
        outputs = local_op(*scan_inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        # Select only the result of the R_op results
        final_outs = []
        b = info.n_mit_mot
        e = info.n_mit_mot * 2
        final_outs += outputs[b:e]
        b = e + info.n_mit_sot
        e = e + info.n_mit_sot * 2
        final_outs += outputs[b:e]
        b = e + info.n_sit_sot
        e = e + info.n_sit_sot * 2
        final_outs += outputs[b:e]
        b = e + info.n_nit_sot
        e = e + info.n_nit_sot * 2
        final_outs += outputs[b:e]
        final_outs += [None] * info.n_shared_outs

        return final_outs


@register_profiler_printer
def profile_printer(
    message, compile_time, fct_call_time, apply_time, apply_cimpl, outputs_size, file
):
    # Scan overhead profile
    if any(
        isinstance(node.op, Scan) and v > 0 for (fgraph, node), v in apply_time.items()
    ):
        print("", file=file)
        print("Scan overhead:", file=file)
        print(
            "<Scan op time(s)> <sub scan fct time(s)> <sub scan op "
            "time(s)> <sub scan fct time(% scan op time)> <sub scan "
            "op time(% scan op time)> <node>",
            file=file,
        )

        total_super_scan_time = 0
        total_scan_fct_time = 0
        total_scan_op_time = 0
        for (fgraph, node), v in apply_time.items():
            if isinstance(node.op, Scan) and not node.op.fn.profile:
                print(
                    "  One scan node do not have its inner profile enabled. "
                    "If you enable Aesara profiler with "
                    "'aesara.function(..., profile=True)', you must manually"
                    " enable the profiling for each scan too: "
                    "'aesara.scan(...,profile=True)'."
                    " Or use Aesara flag 'profile=True'.",
                    file=file,
                )
            elif isinstance(node.op, Scan) and node.op.fn.profile:
                if v > 0:
                    scan_fct_time = node.op.fn.profile.call_time
                    scan_op_time = sum(node.op.fn.profile.apply_time.values())
                    total_super_scan_time += v
                    total_scan_fct_time += scan_fct_time
                    total_scan_op_time += scan_op_time
                    print(
                        "      %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%"
                        % (
                            v,
                            scan_fct_time,
                            scan_op_time,
                            scan_fct_time / v * 100,
                            scan_op_time / v * 100,
                        ),
                        node,
                        file=file,
                    )
                else:
                    print(
                        (" The node took 0s, so we can not " "compute the overhead"),
                        node,
                        file=file,
                    )
        if total_super_scan_time == 0:
            print("  No scan have its inner profile enabled.", file=file)
        else:
            print(
                "total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%"
                % (
                    total_super_scan_time,
                    total_scan_fct_time,
                    total_scan_op_time,
                    total_scan_fct_time / total_super_scan_time * 100,
                    total_scan_op_time / total_super_scan_time * 100,
                ),
                file=file,
            )


@op_debug_information.register(Scan)  # type: ignore[has-type]
def _op_debug_information_Scan(op, node):
    from typing import Sequence

    from aesara.scan.utils import ScanArgs

    extra_information = {}

    inner_fn = getattr(op, "_fn", None)

    if inner_fn:
        inner_inputs = inner_fn.maker.fgraph.inputs
        inner_outputs = inner_fn.maker.fgraph.outputs
    else:
        inner_inputs = op.inner_inputs
        inner_outputs = op.inner_outputs

    scan_args = ScanArgs(
        node.inputs,
        node.outputs,
        inner_inputs,
        inner_outputs,
        node.op.info,
        clone=False,
    )

    for field_name in scan_args.field_names:
        field_vars = getattr(scan_args, field_name)
        if isinstance(field_vars, Sequence):
            for i, var in enumerate(field_vars):
                if isinstance(var, Sequence):
                    for j, sub_var in enumerate(var):
                        extra_information[sub_var] = f"{field_name}-{i}-{j}"
                else:
                    extra_information[var] = f"{field_name}-{i}"
        else:
            extra_information[field_vars] = field_name
    return {node: extra_information}

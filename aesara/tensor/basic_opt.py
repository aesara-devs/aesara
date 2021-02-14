""" Tensor optimizations addressing the ops in basic.py."""

import logging
import sys
import time
import traceback
from collections import defaultdict
from io import StringIO
from typing import Optional

import numpy as np

import aesara
import aesara.scalar.basic as aes
from aesara import compile
from aesara.compile.ops import ViewOp
from aesara.configdefaults import config
from aesara.graph import features
from aesara.graph.basic import (
    Constant,
    Variable,
    ancestors,
    equal_computations,
    io_toposort,
)
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.graph.opt import (
    GlobalOptimizer,
    OpRemove,
    check_chain,
    copy_stack_trace,
    in2out,
    local_optimizer,
)
from aesara.graph.optdb import SequenceDB
from aesara.graph.utils import (
    InconsistencyError,
    MethodNotDefined,
    TestValueError,
    get_variable_trace_string,
)
from aesara.printing import Printer, pprint, set_precedence
from aesara.raise_op import Assert, CheckAndRaise, assert_op
from aesara.tensor.basic import (
    Alloc,
    AllocEmpty,
    Flatten,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    Tile,
    alloc,
    as_tensor_variable,
    cast,
    constant,
    extract_constant,
    fill,
    get_scalar_constant_value,
    get_vector_length,
    join,
    ones_like,
    patternbroadcast,
    switch,
    tensor_copy,
    unbroadcast,
    zeros,
    zeros_like,
)
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError, ShapeError
from aesara.tensor.extra_ops import BroadcastTo, Repeat, Unique, broadcast_shape
from aesara.tensor.math import all as at_all
from aesara.tensor.math import eq
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, shape_padleft
from aesara.tensor.sort import TopKOp
from aesara.tensor.subtensor import Subtensor, get_idx_list
from aesara.tensor.type import (
    DenseTensorType,
    TensorType,
    discrete_dtypes,
    integer_dtypes,
)
from aesara.tensor.var import TensorConstant
from aesara.utils import NoDuplicateOptWarningFilter


_logger = logging.getLogger("aesara.tensor.basic_opt")
_logger.addFilter(NoDuplicateOptWarningFilter())


def encompasses_broadcastable(b1, b2):
    """

    Parameters
    ----------
    b1
        The broadcastable attribute of a tensor type.
    b2
        The broadcastable attribute of a tensor type.

    Returns
    -------
    bool
        True if the broadcastable patterns b1 and b2 are such that b2 is
        broadcasted to b1's shape and not the opposite.

    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2) :]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))


def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]


def broadcast_like(value, template, fgraph, dtype=None):
    """
    Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be cast as
    necessary.

    """
    value = as_tensor_variable(value)
    if value.type.is_super(template.type):
        return value
    if template not in fgraph.variables:
        raise NotImplementedError(
            "broadcast_like currently requires the "
            "template Variable to be in the fgraph already"
        )
    if dtype is None:
        dtype = template.dtype
    value = cast(value, dtype)
    if value.type.is_super(template.type):
        return value
    if hasattr(fgraph, "shape_feature"):
        new_shape = fgraph.shape_feature.shape_of[template]
    else:
        new_shape = template.shape
    rval = alloc(value, *new_shape)
    # the template may have 1s in its shape without being broadcastable
    if rval.broadcastable != template.broadcastable:
        rval = unbroadcast(
            rval,
            *[
                i
                for i in range(rval.ndim)
                if rval.broadcastable[i] and not template.broadcastable[i]
            ],
        )
    assert rval.type.dtype == dtype

    if rval.type.broadcastable != template.broadcastable:
        raise AssertionError(
            "rval.type.broadcastable is "
            + str(rval.type.broadcastable)
            + " but template.broadcastable is"
            + str(template.broadcastable)
        )

    return rval


class InplaceElemwiseOptimizer(GlobalOptimizer):
    r"""
    This is parameterized so that it works for `Elemwise` and `GpuElemwise` `Op`\s.
    """

    def __init__(self, OP):
        self.op = OP

    def add_requirements(self, fgraph):
        from aesara.graph.destroyhandler import DestroyHandler

        fgraph.attach_feature(DestroyHandler())

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "InplaceElemwiseOptimizer ", prof["opt"].op, file=stream)
        for k in [
            "node_before",
            "nb_call_replace",
            "nb_call_validate",
            "nb_inconsistent",
        ]:
            print(blanc, k, prof[k], file=stream)
        ndim = prof["ndim"]
        if ndim:
            print(blanc, "ndim", "nb", file=stream)
            for n in sorted(ndim.keys()):
                print(blanc, n, ndim[n], file=stream)

    def apply(self, fgraph):
        """
        Usage: InplaceElemwiseOptimizer(op).optimize(fgraph)

        Attempts to replace all Broadcast ops by versions of them
        that operate inplace. It operates greedily: for each Broadcast
        Op that is encountered, for each output, tries each input to
        see if it can operate inplace on that input. If so, makes the
        change and go to the next output or Broadcast Op.

        Examples
        --------

            `x + y + z -> x += y += z`

            `(x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)`

        """
        # We should not validate too often as this takes too much time to
        # execute!
        # It is the _dfs_toposort() fct in aesara/graph/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe Aesara should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest optimizer is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # We execute `validate` after this number of change.
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_call_replace": 0,
            "nb_call_validate": 0,
            "nb_inconsistent": 0,
            "ndim": defaultdict(lambda: 0),
        }

        check_each_change = config.tensor__insert_inplace_optimizer_validate_nb
        if check_each_change == -1:
            if len(fgraph.apply_nodes) > 500:
                check_each_change = 10
            else:
                check_each_change = 1

        nb_change_no_validate = 0
        chk = fgraph.checkpoint()

        if fgraph.update_mapping:
            update_outs = [fgraph.outputs[i] for i in fgraph.update_mapping]
        else:
            update_outs = []

        protected_inputs = [
            f.protected
            for f in fgraph._features
            if isinstance(f, aesara.compile.function.types.Supervisor)
        ]
        protected_inputs = sum(protected_inputs, [])  # flatten the list
        protected_inputs.extend(fgraph.outputs)
        for node in list(io_toposort(fgraph.inputs, fgraph.outputs)):
            op = node.op
            # gpuarray GpuElemwise inherit from Elemwise
            if not isinstance(op, self.op):
                continue
            # If big graph and the outputs are scalar, do not make it
            # inplace.
            if (
                check_each_change != 1
                and
                # If multiple outputs, they must all have the same size,
                # so only check the first.
                getattr(node.outputs[0].type, "ndim", -1) == 0
            ):
                continue

            if op.inplace_pattern:
                # Maybe this isn't needed anymore, but I don't want to
                # rish regression now. This case only happen if the
                # original node add already some inplace patter and we
                # still try to add more pattern.

                baseline = op.inplace_pattern
                candidate_outputs = [
                    i for i in range(len(node.outputs)) if i not in baseline
                ]
                # node inputs that are Constant, already destroyed,
                # or fgraph protected inputs and fgraph outputs can't be used as
                # inplace target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if i not in baseline.values()
                    and not isinstance(node.inputs[i], Constant)
                    and
                    # the next line should not be costly most of the time.
                    not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]
            else:
                baseline = []
                candidate_outputs = list(range(len(node.outputs)))
                # node inputs that are Constant, already destroyed,
                # fgraph protected inputs and fgraph outputs can't be used as inplace
                # target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if not isinstance(node.inputs[i], Constant)
                    and not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]

            verbose = False

            raised_warning = not verbose

            for candidate_output in candidate_outputs:

                # If the output of the node can be established as an update
                # output of the fgraph, visit the candidate_inputs in an order
                # that will improve the chances of making the node operate
                # inplace on the input it's meant to update
                candidate_out_var = node.outputs[candidate_output]
                sorted_candidate_inputs = candidate_inputs

                if candidate_out_var in update_outs:

                    # The candidate output is an update. Sort the
                    # variables in candidate_inputs in the following order:
                    # - Vars corresponding to the actual updated input
                    #   (best case scenario is for the node that procudes
                    #   an update to operate inplace on the variable to
                    #   update)
                    # - Vars computed inplace on the updates input (second
                    #   best scenario if for the node to work inplace on
                    #   a variable obtained by a chain of inplace on the
                    #   variable to update. In some cases, this will be
                    #   equivalent to operating inplace on the variable to
                    #   update)
                    # - Remaining variables
                    updated_inputs = []
                    for i, f_out in enumerate(fgraph.outputs):
                        if f_out is candidate_out_var and i in fgraph.update_mapping:
                            updated_inp_idx = fgraph.update_mapping[i]
                            updated_inputs.append(fgraph.inputs[updated_inp_idx])

                    updated_vars = []
                    vars_from_inplace = []
                    other_vars = []
                    for inp_idx in candidate_inputs:
                        inp = node.inputs[inp_idx]
                        if inp in updated_inputs:
                            # the candidate input is the actual updated input
                            updated_vars.append(inp_idx)
                        elif (
                            hasattr(fgraph, "destroy_handler")
                            and inp.owner
                            and any(
                                [
                                    fgraph.destroy_handler.root_destroyer.get(
                                        up_inp, None
                                    )
                                    is inp.owner
                                    for up_inp in updated_inputs
                                ]
                            )
                        ):

                            # the candidate input is a variable computed
                            # inplace on the updated input via a sequence of
                            # one or more inplace operations
                            vars_from_inplace.append(inp_idx)
                        else:
                            other_vars.append(inp_idx)

                    sorted_candidate_inputs = (
                        updated_vars + vars_from_inplace + other_vars
                    )

                for candidate_input in sorted_candidate_inputs:
                    # remove inputs that don't have the same dtype as the output
                    if (
                        node.inputs[candidate_input].type
                        != node.outputs[candidate_output].type
                    ):
                        continue

                    inplace_pattern = dict(baseline)
                    inplace_pattern[candidate_output] = candidate_input
                    try:
                        if hasattr(op.scalar_op, "make_new_inplace"):
                            new_scal = op.scalar_op.make_new_inplace(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, o.dtype)
                                        for i, o in enumerate(node.outputs)
                                    ]
                                )
                            )
                        else:
                            new_scal = op.scalar_op.__class__(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, None)
                                        for i in range(len(node.outputs))
                                    ]
                                )
                            )
                        new_outputs = self.op(new_scal, inplace_pattern)(
                            *node.inputs, return_list=True
                        )
                        new_node = new_outputs[0].owner

                        for r, new_r in zip(node.outputs, new_outputs):
                            prof["nb_call_replace"] += 1
                            fgraph.replace(
                                r, new_r, reason="inplace_elemwise_optimizer"
                            )
                        nb_change_no_validate += 1
                        prof["ndim"][candidate_out_var.ndim] += 1
                        if nb_change_no_validate >= check_each_change:
                            prof["nb_call_validate"] += 1
                            fgraph.validate()
                            chk = fgraph.checkpoint()
                            nb_change_no_validate = 0
                    except (ValueError, InconsistencyError) as e:
                        prof["nb_inconsistent"] += 1
                        if check_each_change != 1 and not raised_warning:
                            print(
                                (
                                    "Some inplace optimization was not "
                                    "performed due to unexpected error:"
                                ),
                                file=sys.stderr,
                            )
                            print(e, file=sys.stderr)
                            raised_warning = True
                        fgraph.revert(chk)
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new_node
                    baseline = inplace_pattern
                    break

        if nb_change_no_validate > 0:
            try:
                fgraph.validate()
            except Exception:
                if not raised_warning:
                    print(
                        (
                            "Some inplace optimization was not "
                            "performed due to unexpected error"
                        ),
                        file=sys.stderr,
                    )
                fgraph.revert(chk)
        return prof

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__} ({self.op})",
            file=stream,
        )
        return inplace_elemwise_optimizer


inplace_elemwise_optimizer = InplaceElemwiseOptimizer(Elemwise)
compile.optdb.register(
    "inplace_elemwise_opt",
    inplace_elemwise_optimizer,
    "inplace_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    position=75,
)


def register_useless(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_useless(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__

        compile.mode.local_useless.register(
            name, lopt, "fast_run", *tags, position="last", **kwargs
        )
        return lopt


def register_canonicalize(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_canonicalize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["canonicalize"].register(
            name, lopt, "fast_run", "fast_compile", *tags, **kwargs
        )
        return lopt


def register_stabilize(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_stabilize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["stabilize"].register(name, lopt, "fast_run", *tags, **kwargs)
        return lopt


def register_specialize(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_specialize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["specialize"].register(name, lopt, "fast_run", *tags, **kwargs)
        return lopt


def register_uncanonicalize(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_uncanonicalize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or lopt.__name__
        compile.optdb["uncanonicalize"].register(
            name, lopt, "fast_run", *tags, **kwargs
        )
        return lopt


def register_specialize_device(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_specialize_device(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or lopt.__name__
        compile.optdb["specialize_device"].register(
            name, lopt, "fast_run", *tags, **kwargs
        )
        return lopt


def apply_local_dimshuffle_lift(fgraph, var):
    """
    lift recursively
    """
    if not var.owner:
        return var
    new = local_dimshuffle_lift.transform(fgraph, var.owner)
    if new:
        return new[0]
    return var


def is_dimshuffle_useless(new_order, input):
    """
    Checks for two types of useless dimshuffles:
      1 - dimshuffle all dimensions in order.
      2 - dimshuffle a broadcastable dimension.
    """
    is_useless = True
    if len(new_order) == input.type.ndim:
        all_broadcastable_dims = [
            i
            for (i, is_broadcastable) in enumerate(input.type.broadcastable)
            if is_broadcastable
        ] + ["x"]
        for i in range(input.type.ndim):
            if new_order[i] == i or (
                i in all_broadcastable_dims and new_order[i] in all_broadcastable_dims
            ):
                is_useless = True
            else:
                is_useless = False
                break
    else:
        is_useless = False
    return is_useless


@register_canonicalize
@register_specialize
@local_optimizer([DimShuffle])
def local_dimshuffle_lift(fgraph, node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)
    DimShuffle{0,1,...}(x) => x (when the dimshuffle do nothing)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.

    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    inp = node.inputs[0]
    inode = inp.owner
    new_order = op.new_order
    if inode and isinstance(inode.op, Elemwise) and (len(fgraph.clients[inp]) == 1):
        # Don't use make_node to have tag.test_value set.
        new_inputs = []
        for inp in inode.inputs:
            new_inp = op.__class__(inp.type.broadcastable, op.new_order)(inp)
            new_inputs.append(apply_local_dimshuffle_lift(fgraph, new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, return_list=True)
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == "x" and "x" or inode.op.new_order[x] for x in new_order]
        inp = inode.inputs[0]

    if is_dimshuffle_useless(new_order, inp):
        return [inp]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = op.__class__(inp.type.broadcastable, new_order)(inp)
        ret = apply_local_dimshuffle_lift(fgraph, ret)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@register_specialize
@local_optimizer([DimShuffle])
def local_useless_dimshuffle_makevector(fgraph, node):
    r"""Remove `DimShuffle`\s that drop one dimensional broadcastable `MakeVector`s.

    This rewrite is needed in order to clean up after
    `local_subtensor_remove_broadcastable_index`, which produces a
    not-so-intuitive canonical form for `x[0]` when `x.shape == (1,)`
    (i.e. one broadcastable dimension): i.e. `x.dimshuffle(())`.
    """

    # The `DimShuffle` should be removing the single broadcastable dimension
    if node.op.new_order != ():
        return

    makevector_out = node.inputs[0]

    if (
        not makevector_out.owner
        or not isinstance(makevector_out.owner.op, MakeVector)
        or not makevector_out.broadcastable == (True,)
    ):
        return

    assert len(makevector_out.owner.inputs) == 1

    return [makevector_out.owner.inputs[0]]


@register_canonicalize
@local_optimizer([Reshape])
def local_useless_dimshuffle_in_reshape(fgraph, node):
    """
    Removes useless DimShuffle operation inside Reshape:

      reshape(vector.dimshuffle('x', 0), shp) => reshape(vector, shp)
      reshape(matrix.dimshuffle('x', 0, 'x', 1), shp) => reshape(matrix, shp)
      reshape(row.dimshuffle(1, 'x'), shp) => reshape(row, shp)
      reshape(col.dimshuffle(0), shp) => reshape(col, shp)

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False
    if not (
        node.inputs[0].owner is not None
        and isinstance(node.inputs[0].owner.op, DimShuffle)
    ):
        return False

    new_order = node.inputs[0].owner.op.new_order
    inp = node.inputs[0].owner.inputs[0]
    broadcastables = node.inputs[0].broadcastable
    new_order_of_nonbroadcast = []
    for i, bd in zip(new_order, broadcastables):
        if not bd:
            new_order_of_nonbroadcast.append(i)
    no_change_in_order = all(
        new_order_of_nonbroadcast[i] <= new_order_of_nonbroadcast[i + 1]
        for i in range(len(new_order_of_nonbroadcast) - 1)
    )
    if no_change_in_order:
        shape = node.inputs[1]
        ret = op.__class__(node.outputs[0].ndim)(inp, shape)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@register_specialize
@local_optimizer([TensorFromScalar])
def local_tensor_scalar_tensor(fgraph, node):
    """tensor_from_scalar(scalar_from_tensor(x)) -> x"""
    if isinstance(node.op, TensorFromScalar):
        s = node.inputs[0]
        if s.owner and isinstance(s.owner.op, ScalarFromTensor):
            t = s.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [t]


@register_canonicalize
@register_specialize
@local_optimizer([ScalarFromTensor])
def local_scalar_tensor_scalar(fgraph, node):
    """scalar_from_tensor(tensor_from_scalar(x)) -> x"""
    if isinstance(node.op, ScalarFromTensor):
        t = node.inputs[0]
        if t.owner and isinstance(t.owner.op, TensorFromScalar):
            s = t.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [s]


class MakeVectorPrinter(Printer):
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            with set_precedence(pstate):
                s = [pstate.pprinter.process(inp) for inp in r.owner.inputs]
            return f"[{', '.join(s)}]"
        else:
            raise TypeError("Can only print make_vector.")


pprint.assign(MakeVector, MakeVectorPrinter())


class ShapeFeature(features.Feature):
    """Graph optimizer for removing all calls to shape().

    This optimizer replaces all Shapes and Subtensors of Shapes with
    Shape_i and MakeVector Ops.

    This optimizer has several goals:

    1. to 'lift' Shapes to as close to the inputs as possible.

    2. to infer the shape of every node in the graph in terms of the
       input shapes.

    3. remove all fills ``(at.second, at.fill)`` from the graph

    Lifting shapes as close to the inputs as possible is important for
    canonicalization because it is very bad form to have to compute
    something just to know how big it will be.  Firstly, it is a waste
    of time to compute such outputs.  But it is important to get rid
    of these outputs as early as possible in the compilation process
    because the extra computations make it appear as if many internal
    graph nodes have multiple clients.  Many optimizations refuse to
    work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is
    present, or else using a conservative default.  An Op that
    supports shape-lifting should define a infer_shape(self, fgraph, node,
    input_shapes) function.  The argument input_shapes is a tuple of
    tuples... there is an interior tuple for each input to the node.
    The tuple has as many elements as dimensions.  The element in
    position i of tuple j represents the i'th shape component of the
    j'th input.  The function should return a tuple of tuples.  One
    output tuple for each node.output.  Again, the i'th element of the
    j'th output tuple represents the output[j].shape[i] of the
    function.  If an output is not a TensorType, then None should be
    returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).

    Inferring the shape of internal nodes in the graph is important
    for doing size-driven optimizations.  If we know how big various
    intermediate results will be, we can estimate the cost of many Ops
    accurately, and generate c-code that is specific [e.g. unrolled]
    to particular sizes.

    In cases where you cannot figure out the shape, raise a ShapeError.

    Notes
    -----
    Right now there is only the ConvOp that could really take
    advantage of this shape inference, but it is worth it even
    just for the ConvOp.  All that's necessary to do shape
    inference is 1) to mark shared inputs as having a particular
    shape, either via a .tag or some similar hacking; and 2) to
    add an optional In() argument to promise that inputs will
    have a certain shape (or even to have certain shapes in
    certain dimensions). We can't automatically infer the shape of
    shared variables as they can change of shape during the
    execution by default.  (NOT IMPLEMENTED YET, BUT IS IN TRAC)


    **Using Shape information in Optimizations**

    To use this shape information in OPTIMIZATIONS, use the
    ``shape_of`` dictionary.

    For example:

    .. code-block:: python

        try:
            shape_of = fgraph.shape_feature.shape_of
        except AttributeError:
            # This can happen when the mode doesn't include the ShapeFeature.
            return

        shape_of_output_zero = shape_of[node.output[0]]

    The ``shape_of_output_zero`` symbol will contain a tuple, whose
    elements are either integers or symbolic integers.

    TODO: check to see if the symbols are necessarily
    non-constant... or are integer literals sometimes Aesara
    constants?? That would be confusing.

    """

    def get_node_infer_shape(self, node):
        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except ShapeError:
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Code called by infer_shape failed raising a "
                "NotImplementedError. Raising NotImplementedError to "
                "indicate that a shape cannot be computed is no longer "
                "supported, and one should now use ShapeError "
                f"instead. The original exception message is: {e}"
            ).with_traceback(e.__traceback__)
        except Exception as e:
            msg = (
                f"Failed to infer_shape from Op {node.op}.\nInput shapes: "
                f"{[self.shape_of[r] for r in node.inputs]}\nException encountered during infer_shape: "
                f"{type(e)}\nException message: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            if config.on_shape_error == "raise":
                raise Exception(msg).with_traceback(e.__traceback__)
            else:
                _logger.warning(msg)
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )

        return o_shapes

    def get_shape(self, var, idx):
        """Optimization can call this to get the current shape_i

        It is better to call this then use directly shape_of[var][idx]
        as this method should update shape_of if needed.

        TODO: Up to now, we don't update it in all cases. Update in all cases.
        """
        r = self.shape_of[var][idx]
        if (
            r.owner
            and isinstance(r.owner.op, Shape_i)
            and r.owner.inputs[0] not in self.fgraph.variables
        ):
            assert var.owner
            node = var.owner
            # recur on inputs
            for i in node.inputs:
                if getattr(i.type, "ndim", None) > 0:
                    self.get_shape(i, 0)
            o_shapes = self.get_node_infer_shape(node)
            assert len(o_shapes) == len(node.outputs)

            # Only change the variables and dimensions that would introduce
            # extra computation
            for new_shps, out in zip(o_shapes, node.outputs):
                if not hasattr(out.type, "ndim"):
                    continue

                merged_shps = list(self.shape_of[out])
                changed = False
                for i in range(out.type.ndim):
                    n_r = merged_shps[i]
                    if (
                        n_r.owner
                        and isinstance(n_r.owner.op, Shape_i)
                        and n_r.owner.inputs[0] not in self.fgraph.variables
                    ):
                        changed = True
                        merged_shps[i] = new_shps[i]
                if changed:
                    self.set_shape(out, merged_shps, override=True)
            r = self.shape_of[var][idx]
        return r

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i."""
        if hasattr(r.type, "shape") and r.type.shape[i] is not None:
            return constant(r.type.shape[i], dtype="int64")
        else:
            # Do not call make_node for test_value
            s = Shape_i(i)(r)
            try:
                s = get_scalar_constant_value(s)
            except NotScalarConstantError:
                pass
            return s

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r.type, "ndim"):
            # This happen for NoneConst.
            return None
        return tuple(self.shape_ir(i, r) for i in range(r.type.ndim))

    def default_infer_shape(self, fgraph, node, i_shapes):
        """Return a list of shape tuple or None for the outputs of node.

        This function is used for Ops that don't implement infer_shape.
        Ops that do implement infer_shape should use the i_shapes parameter,
        but this default implementation ignores it.

        """
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i, var):
        """Return a symbolic integer scalar for the shape element s_i.

        The s_i argument was produced by the infer_shape() of an Op subclass.

        var: the variable that correspond to s_i. This is just for
        error reporting.

        """
        # unpack the s_i that the Op returned
        assert s_i is not None
        if s_i == 1:
            # don't make the optimizer merge a zillion ones together
            # by always returning the same object to represent 1
            return self.lscalar_one
        if isinstance(s_i, float) and int(s_i) == s_i:
            s_i = int(s_i)
        if isinstance(s_i, (np.integer, int)) or (
            isinstance(s_i, np.ndarray) and s_i.ndim == 0
        ):
            # this shape is a constant
            if s_i < 0:
                msg = "There is a negative shape in the graph!"
                msg += get_variable_trace_string(var)
                # The rest of the pipeline don't handle correctly this
                # case.  So we have 2 choices, stop compilation or
                # consider the shape as unknown.  As we have more
                # chance to give the stack trace here then later, I
                # choose that options as it would give better error
                # message.
                raise AssertionError(msg)
            return constant(s_i, dtype="int64")
        if isinstance(s_i, (tuple, list)):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known,
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)

        # s_i is x.shape[i] for some x, we change it to shape_of[x][i]
        if (
            s_i.owner
            and isinstance(s_i.owner.op, Subtensor)
            and s_i.owner.inputs[0].owner
            and isinstance(s_i.owner.inputs[0].owner.op, Shape)
        ):
            assert s_i.type.ndim == 0
            assert len(s_i.owner.op.idx_list) == 1

            # The current Subtensor always put constant index in the graph.
            # This was not True in the past. So call the Subtensor function
            # that will return the right index.
            idx = get_idx_list(s_i.owner.inputs, s_i.owner.op.idx_list)
            assert len(idx) == 1
            idx = idx[0]
            try:
                i = get_scalar_constant_value(idx)
            except NotScalarConstantError:
                pass
            else:
                # Executed only if no exception was raised
                x = s_i.owner.inputs[0].owner.inputs[0]
                # x should already have been imported, and should be in shape_of.
                s_i = self.shape_of[x][i]

        if s_i.type.dtype in integer_dtypes:
            if getattr(s_i.type, "ndim", 0):
                raise TypeError("Shape element must be scalar", s_i)
            return s_i
        else:
            raise TypeError(
                "Unsupported shape element", s_i, type(s_i), getattr(s_i, "type", None)
            )

    def set_shape(self, r, s, override=False):
        """Assign the shape `s` to previously un-shaped variable `r`.

        Parameters
        ----------
        r : a variable
        s : None or a tuple of symbolic integers
        override : If False, it mean r is a new object in the fgraph.
            If True, it mean r is already in the fgraph and we want to
            override its shape.

        """
        if not override:
            assert r not in self.shape_of, "r already in shape_of"
        if s is None:
            self.shape_of[r] = s
        else:
            if not isinstance(s, (tuple, list)):
                raise TypeError("shapes must be tuple/list", (r, s))

            if r.type.ndim != len(s):
                sio = StringIO()
                aesara.printing.debugprint(r, file=sio, print_type=True)
                raise AssertionError(
                    f"Something inferred a shape with {len(s)} dimensions "
                    f"for a variable with {int(r.type.ndim)} dimensions"
                    f" for the variable:\n{sio.getvalue()}"
                )

            shape_vars = []
            for i in range(r.type.ndim):
                if hasattr(r.type, "shape") and r.type.shape[i] is not None:
                    shape_vars.append(constant(r.type.shape[i], dtype="int64"))
                else:
                    shape_vars.append(self.unpack(s[i], r))
            assert all(
                not hasattr(r.type, "broadcastable") or not r.type.broadcastable[i] or
                # The two following comparison are a speed optimization
                # But we never timed this speed optimization!
                self.lscalar_one.equals(shape_vars[i])
                or self.lscalar_one.equals(extract_constant(shape_vars[i]))
                for i in range(r.type.ndim)
            )
            self.shape_of[r] = tuple(shape_vars)
            for sv in shape_vars:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def update_shape(self, r, other_r):
        """Replace shape of r by shape of other_r.

        If, on some dimensions, the shape of other_r is not informative,
        keep the shape of r on those dimensions.

        """
        # other_r should already have a shape
        assert other_r in self.shape_of, ("other_r not in shape_of", other_r)
        other_shape = self.shape_of[other_r]

        # If other_shape has no information, call is pointless.
        if other_shape is None:
            return

        if r in self.shape_of:
            r_shape = self.shape_of[r]
        else:
            # If no info is known on r's shape, use other_shape
            self.set_shape(r, other_shape)
            return
        if (
            other_r.owner
            and r.owner
            and other_r.owner.inputs == r.owner.inputs
            and other_r.owner.op == r.owner.op
        ):
            # We are doing a merge. So the 2 shapes graph will be the
            # same.  This is only a speed optimization to call
            # ancestors() less frequently.
            return

        # Merge other_shape with r_shape, giving the priority to other_shape
        merged_shape = []
        for i, ps in enumerate(other_shape):
            if r_shape is None and other_shape:
                merged_shape.append(other_shape[i])
            elif (
                ps.owner
                and isinstance(getattr(ps.owner, "op", None), Shape_i)
                and ps.owner.op.i == i
                and ps.owner.inputs[0] in (r, other_r)
            ):
                # If other_shape[i] is uninformative, use r_shape[i].
                # For now, we consider 2 cases of uninformative other_shape[i]:
                #  - Shape_i(i)(other_r);
                #  - Shape_i(i)(r).
                merged_shape.append(r_shape[i])
            elif isinstance(r_shape[i], (Constant, int)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(r_shape[i])
            elif isinstance(other_shape[i], (Constant, int)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(other_shape[i])
            elif other_shape[i] == r_shape[i]:
                # This mean the shape is equivalent
                # We do not want to do the ancestor check in those cases
                merged_shape.append(r_shape[i])
            elif r_shape[i] in ancestors([other_shape[i]]):
                # Another case where we want to use r_shape[i] is when
                # other_shape[i] actually depends on r_shape[i]. In that case,
                # we do not want to substitute an expression with another that
                # is strictly more complex. Such a substitution could also lead
                # to cycles: if (in the future) r_shape[i] gets replaced by an
                # expression of other_shape[i], other_shape[i] may end up
                # depending on itself.
                merged_shape.append(r_shape[i])
            else:
                merged_shape.append(other_shape[i])
        assert all(
            (
                not hasattr(r.type, "broadcastable")
                or not r.type.broadcastable[i]
                and not other_r.type.broadcastable[i]
            )
            or
            # The two following comparison are a speed optimization
            # But we never timed this speed optimization!
            self.lscalar_one.equals(merged_shape[i])
            or self.lscalar_one.equals(
                extract_constant(merged_shape[i], only_process_constants=True)
            )
            for i in range(r.type.ndim)
        )
        self.shape_of[r] = tuple(merged_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def set_shape_i(self, r, i, s_i):
        """Replace element i of shape_of[r] by s_i"""
        assert r in self.shape_of
        prev_shape = self.shape_of[r]
        # prev_shape is a tuple, so we cannot change it inplace,
        # so we build another one.
        new_shape = []
        for j, s_j in enumerate(prev_shape):
            if j == i:
                new_shape.append(self.unpack(s_i, r))
            else:
                new_shape.append(s_j)
        assert all(
            not hasattr(r.type, "broadcastable") or not r.type.broadcastable[idx] or
            # The two following comparison are a speed optimization
            # But we never timed this speed optimization!
            self.lscalar_one.equals(new_shape[idx])
            or self.lscalar_one.equals(extract_constant(new_shape[idx]))
            for idx in range(r.type.ndim)
        )
        self.shape_of[r] = tuple(new_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def init_r(self, r):
        """Register r's shape in the shape_of dictionary."""
        if r not in self.shape_of:
            self.set_shape(r, self.shape_tuple(r))

    def make_vector_shape(self, r):
        return as_tensor_variable(self.shape_of[r], ndim=1, dtype="int64")

    def on_attach(self, fgraph):

        if getattr(self, "fgraph", None):
            raise ValueError("This ShapeFeature is already attached to a graph")

        self.fgraph = fgraph

        if hasattr(fgraph, "shape_feature"):
            raise ValueError("This FunctionGraph already has a ShapeFeature")

        fgraph.shape_feature = self
        # Must be local to the object as otherwise we reuse the same
        # variable for multiple fgraph!
        self.lscalar_one = constant(1, dtype="int64")
        assert self.lscalar_one.type.dtype == "int64"

        self.fgraph = fgraph
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.shape_of = {}
        # Variable ->
        self.scheduled = {}
        # shape var -> graph v
        self.shape_of_reverse_index = {}

        for node in fgraph.toposort():
            self.on_import(fgraph, node, reason="on_attach")

    def on_detach(self, fgraph):
        self.shape_of = {}
        self.scheduled = {}
        self.shape_of_reverse_index = {}
        self.fgraph = None
        del fgraph.shape_feature

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        o_shapes = self.get_node_infer_shape(node)

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints
        if len(o_shapes) != len(node.outputs):
            raise Exception(
                (
                    f'The infer_shape method for the Op "{node.op}" returned a list '
                    f"with the wrong number of element: len(o_shapes) = {len(o_shapes)} "
                    f" != len(node.outputs) = {len(node.outputs)}"
                )
            )

        # Ensure shapes are in 'int64'. This is to make sure the assert
        # found in the `local_useless_subtensor` optimization does not fail.
        for sh_idx, sh in enumerate(o_shapes):
            if sh is None:
                continue
            if not isinstance(sh, (list, tuple)):
                raise ValueError(
                    f"infer_shape of {node} didn't return a list of"
                    f" list. It returned '{o_shapes}'"
                )
            new_shape = []
            for i, d in enumerate(sh):
                # Note: we ignore any shape element that is not typed (i.e.,
                # does not have a 'dtype' attribute). This means there may
                # still remain int elements that are int32 on 32-bit platforms,
                # but this works with `local_useless_subtensor`, so for now we
                # keep it this way. See #266 for a better long-term fix.
                if getattr(d, "dtype", "int64") != "int64":
                    assert d.dtype in discrete_dtypes, (node, d.dtype)
                    assert str(d.dtype) != "uint64", node
                    new_shape += sh[len(new_shape) : i + 1]
                    if isinstance(d, Constant):
                        casted_d = constant(d.data, dtype="int64")
                    else:
                        casted_d = cast(d, "int64")
                    new_shape[i] = casted_d
            if new_shape:
                # We replace the shape with wrong dtype by the one with
                # 'int64'.
                new_shape += sh[len(new_shape) :]
                o_shapes[sh_idx] = tuple(new_shape)

        for r, s in zip(node.outputs, o_shapes):
            self.set_shape(r, s)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if new_r not in self.shape_of:
            # It happen that the fgraph didn't called on_import for some
            # new_r.  This happen when new_r don't have an
            # owner(i.e. it is a constant or an input of the graph)
            # update_shape suppose that r and new_r are in shape_of.
            self.init_r(new_r)

        # This tells us that r and new_r must have the same shape if
        # we didn't know that the shapes are related, now we do.
        self.update_shape(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to
        # replace the shape_i of r with the shape of new_r.  Say that
        # r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for (shpnode, idx) in fgraph.clients[r] + [(node, i)]:
            if isinstance(getattr(shpnode, "op", None), Shape_i):
                idx = shpnode.op.i
                repl = self.shape_of[new_r][idx]
                if repl.owner is shpnode:
                    # This mean the replacement shape object is
                    # exactly the same as the current shape object. So
                    # no need for replacement. This happen for example
                    # with the InputToGpuOptimizer optimizer.
                    continue
                if (
                    repl.owner
                    and repl.owner.inputs[0] is shpnode.inputs[0]
                    and isinstance(repl.owner.op, Shape_i)
                    and repl.owner.op.i == shpnode.op.i
                ):
                    # The replacement is a shape_i of the same
                    # input. So no need to do this equivalent
                    # replacement.
                    continue

                if shpnode.outputs[0] in ancestors([repl]):
                    raise InconsistencyError(
                        "This substitution would insert a cycle in the graph:"
                        f"node: {node}, i: {i}, r: {r}, new_r: {new_r}"
                    )

                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update,
        # then we should cancel it.
        unscheduled = [k for k, v in self.scheduled.items() if v == r]
        for k in unscheduled:
            del self.scheduled[k]

        # In either case, r could be in shape_of.values(), that is, r itself
        # is the shape of  something. In that case, we want to update
        # the value in shape_of, to keep it up-to-date.
        for v in self.shape_of_reverse_index.get(r, []):
            # The reverse index is only approximate. It is not updated on
            # deletion of variables, or on change_input so it might be the
            # case that there are a few extra `v`'s in it that no longer have
            # a shape of r or possibly have been deleted from shape_of
            # entirely. The important thing is that it permits to recall
            # all variables with r in their shape.
            for ii, svi in enumerate(self.shape_of.get(v, [])):
                if svi == r:
                    self.set_shape_i(v, ii, new_r)
        self.shape_of_reverse_index[r] = set()

    def same_shape(
        self,
        x: Variable,
        y: Variable,
        dim_x: Optional[int] = None,
        dim_y: Optional[int] = None,
    ) -> bool:
        """Return ``True`` if `x` and `y` have the same shape.

        Parameters
        ==========
        x
            The `Variable` for which its shape is to be compared with `y`'s shape.
        y
            The `Variable` for which its shape is to be compared with `x`'s shape.
        dim_x
            If non ``None``, compare only the dimension of `x` equal to
            `dim_x`.
        dim_y
            If non ``None``, compare only the dimension of `y` equal to
            `dim_y`.

        """
        sx = self.shape_of[x]
        sy = self.shape_of[y]

        if sx is None or sy is None:
            return False

        if dim_x is not None:
            sx = [sx[dim_x]]

        if dim_y is not None:
            sy = [sy[dim_y]]

        if len(sx) != len(sy):
            return False

        # Canonicalize the graphs so that comparisons are reasonable
        # TODO FIXME: This should *not* need to be performed manually here.
        # Instead, the shape information in `self.shape_of` should be operated
        # upon alongside all the other elements in a `FunctionGraph` (e.g. as
        # if `self.shape_of.values()` were additional outputs).
        shapes_fg = FunctionGraph(
            outputs=sx + sy,
            # features=[self],
            clone=True,
            # copy_inputs=False,
        )
        from aesara.graph.opt_utils import optimize_graph

        canon_shapes = optimize_graph(
            shapes_fg, custom_opt=topo_constant_folding
        ).outputs

        sx = canon_shapes[: len(sx)]
        sy = canon_shapes[len(sx) :]

        for dx, dy in zip(sx, sy):
            if not equal_computations([dx], [dy]):
                return False

        return True


class ShapeOptimizer(GlobalOptimizer):
    """Optimizer that adds `ShapeFeature` as a feature."""

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ShapeFeature())

    def apply(self, fgraph):
        pass


class UnShapeOptimizer(GlobalOptimizer):
    """Optimizer that removes `ShapeFeature` as a feature."""

    def apply(self, fgraph):
        for feature in fgraph._features:
            if isinstance(feature, ShapeFeature):
                fgraph.remove_feature(feature)


# Register it after merge1 optimization at 0. We don't want to track
# the shape of merged node.
aesara.compile.mode.optdb.register(
    "ShapeOpt", ShapeOptimizer(), "fast_run", "fast_compile", position=0.1
)
# Not enabled by default for now. Some crossentropy opt use the
# shape_feature.  They are at step 2.01. uncanonicalize is at step
# 3. After it goes to 48.5 that move to the gpu. So 10 seems reasonable.
aesara.compile.mode.optdb.register("UnShapeOpt", UnShapeOptimizer(), position=10)


@register_specialize("local_alloc_elemwise")
@local_optimizer([Elemwise])
def local_elemwise_alloc(fgraph, node):
    r"""Remove unnecessary `Alloc`\s that occur as inputs of `Elemwise` `Op`\s.

    `Alloc`\s are effectively a type of `Elemwise` operation
    (e.g. ``Elemwise{second}(y, x)`` is the same as ``Alloc(x, *y.shape)``), so
    this rewrite uses that fact to reduce `Elemwise`\s on `Alloc`\s to
    `Elemwise`\s of the `Alloc`\s first/value input (i.e. the value it
    broadcasts).

    In other words, this rewrite causes `Elemwise` `Op`\s to "absorb" redundant
    `Alloc`\s.

    The rewrite essentially performs the following replacement:
    ``Elemwise{op}(..., Alloc(x, s), ..., y, ...) -> Elemwise{op}(..., x, ..., y, ...)``,
    when ``y.shape`` for some input ``y`` (or the combined shapes of the
    non-`Alloc`\s) is sufficient to maintain the same/correct output shape.

    In it's current form, it also explicitly accounts for `DimShuffle`\s of
    `Alloc`\s.  This is largely due to `local_alloc_sink_dimshuffle`, which
    introduces them as a canonicalization of `Alloc`'s with leading
    broadcastable dimensions.
    """
    if not isinstance(node.op, Elemwise):
        return False

    # Rewrite is only applicable when there are at least two inputs
    if len(node.inputs) == 1:
        return None

    if len(node.outputs) > 1:
        # Ensure all outputs have the same broadcast pattern
        # This is a supposition that I'm not sure is always true.
        assert all(
            [
                o.type.broadcastable == node.outputs[0].type.broadcastable
                for o in node.outputs[1:]
            ]
        )

    # The broadcast pattern of the output must match the broadcast
    # pattern of at least one of the inputs.
    if not any(
        [
            i.type.broadcastable == node.outputs[0].type.broadcastable
            for i in node.inputs
        ]
    ):
        return False

    def dimshuffled_alloc(i):
        return (
            isinstance(i.owner.op, DimShuffle)
            and i.owner.inputs[0].owner
            and isinstance(i.owner.inputs[0].owner.op, Alloc)
        )

    # At least one input must have an owner that is either a `Alloc` or a
    # `DimShuffle` with an owner that is a `Alloc` -- otherwise there is
    # nothing to optimize.
    if not any(
        [
            i.owner and (isinstance(i.owner.op, Alloc) or dimshuffled_alloc(i))
            for i in node.inputs
        ]
    ):
        return False

    # Search for a non `Alloc` or `DimShuffle` of `Alloc` input that we can use as a
    # baseline for the dimensions.
    assert_op_idx = None
    for idx, i in enumerate(node.inputs):
        if i.type.broadcastable == node.outputs[0].type.broadcastable:
            # Prefer an input that is not a `Alloc` nor a `DimShuffle` of a
            # `Alloc` so that all `Alloc`s can be optimized.
            if not (
                i.owner and (isinstance(i.owner.op, Alloc) or dimshuffled_alloc(i))
            ):
                assert_op_idx = idx
                break

    # If only `Alloc` and `DimShuffle` of `Alloc` exist, we pick the first suitable one
    if assert_op_idx is None:
        for idx, i in enumerate(node.inputs):
            if (i.type.broadcastable == node.outputs[0].type.broadcastable) and (
                i.owner and (isinstance(i.owner.op, Alloc) or dimshuffled_alloc(i))
            ):
                assert_op_idx = idx
                break

    assert_op_in = node.inputs[assert_op_idx]
    cmp_op = assert_op_in
    new_i = []
    same_shape = fgraph.shape_feature.same_shape
    for i in node.inputs:
        # Remove `Alloc`
        if i.owner and isinstance(i.owner.op, Alloc):
            assert i.type.ndim == cmp_op.ndim
            if config.experimental__local_alloc_elemwise_assert:
                get_shape = fgraph.shape_feature.get_shape
                cond = []
                for idx in range(i.type.ndim):
                    if not i.type.broadcastable[idx] and not same_shape(
                        i, cmp_op, idx, idx
                    ):
                        i_shp = get_shape(i, idx)
                        cmp_shp = get_shape(cmp_op, idx)
                        cond.append(eq(i_shp, cmp_shp))
                if cond:
                    assert_op_in = assert_op(assert_op_in, *cond)
            alloc_input = i.owner.inputs[0]
            if alloc_input.ndim != i.ndim:
                # The `Alloc` can add dimensions to the value.
                # We replace those cases with a `DimShuffle` here.
                nb_dim_to_add = i.ndim - alloc_input.ndim
                alloc_input = alloc_input.dimshuffle(
                    ["x"] * nb_dim_to_add + list(range(alloc_input.ndim))
                )
            copy_stack_trace(i, alloc_input)
            new_i.append(alloc_input)

        # Remove `Alloc` in `DimShuffle`
        elif i.owner and dimshuffled_alloc(i):
            assert i.type.ndim == cmp_op.type.ndim
            if config.experimental__local_alloc_elemwise_assert:
                assert_cond = [
                    eq(i.shape[idx], cmp_op.shape[idx])
                    for idx in range(i.type.ndim)
                    if not i.type.broadcastable[idx]
                    and not same_shape(i, cmp_op, idx, idx)
                ]
                if assert_cond:
                    assert_op_in = assert_op(assert_op_in, *assert_cond)
            alloc_input = i.owner.inputs[0].owner.inputs[0]
            if alloc_input.ndim != i.owner.inputs[0].ndim:
                # The `Alloc` can add dimensions to the value.
                # We replace those cases with a `DimShuffle` here.
                # We let later optimizations merge the nested `DimShuffle`s
                nb_dim_to_add = i.owner.inputs[0].ndim - alloc_input.ndim
                alloc_input = alloc_input.dimshuffle(
                    ["x"] * nb_dim_to_add + list(range(alloc_input.ndim))
                )

            # We need to keep the old `DimShuffle`. It could swap axes or
            # add dimensions anywhere.
            r_i = i.owner.op(alloc_input)
            copy_stack_trace(i, r_i)
            new_i.append(r_i)

        else:
            new_i.append(i)
    new_i[assert_op_idx] = assert_op_in

    # If this assert is triggered, it means we are recreating an equivalent graph
    # which would result in a cyclical merge optimization.
    if all(new is old for new, old in zip(new_i, node.inputs)):
        return

    ret = node.op(*new_i, return_list=True)
    copy_stack_trace(node.outputs, ret)
    return ret


@register_canonicalize
@local_optimizer([Elemwise])
def local_fill_sink(fgraph, node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(c, fill(a, f(b, d, e)))
    f need to be an elemwise that isn't a fill.
    """
    if not hasattr(node, "op") or not isinstance(node.op, Elemwise) or node.op == fill:
        return False
    models = []
    inputs = []
    for inp in node.inputs:
        if inp.owner and inp.owner.op == fill:
            models.append(inp.owner.inputs[0])
            inputs.append(inp.owner.inputs[1])
        else:
            inputs.append(inp)
    if not models:
        return False
    c = node.op(*inputs)
    for model in models:
        if (
            model.type.dtype != c.type.dtype
            or model.type.broadcastable != c.type.broadcastable
        ):
            c = fill(model, c)

    # The newly created node c doesn't has 'clients',
    # so this iteration is took place with node.outputs[0]
    replacements = {node.outputs[0]: c}
    for client, cl_idx in fgraph.clients[node.outputs[0]]:
        if (
            hasattr(client, "op")
            and isinstance(client.op, Elemwise)
            and client.op != fill
        ):
            client_inputs = client.inputs[:]
            client_inputs[cl_idx] = c
            new_client = client.op(*client_inputs)

            # Add clients to new_client
            fgraph.clients[new_client.owner.outputs[0]] = fgraph.clients[
                client.outputs[0]
            ]
            r = local_fill_sink.transform(fgraph, new_client.owner)
            if not r:
                continue
            replacements.update(r)
    return replacements


@register_specialize
@register_stabilize
@local_optimizer([fill])
def local_fill_to_alloc(fgraph, node):
    r"""Remove `fill`\s or replace them with `Alloc`\s.

    `Alloc`\s are preferable because they replace explicit tensor dependencies
    with their dependencies on those tensors' shapes, and sometimes those
    shapes can be computed without needing to compute the tensors themselves.

    XXX: This rewrite can produce inconsistent results, so do *not* consider
    making it a canonicalization until those inconsistencies are
    resolved/justified.
    """
    shape_ref, values_ref = node.inputs
    out_type = node.outputs[0].type

    if values_ref.type.broadcastable == out_type.broadcastable:
        # The assumption here is that `values_ref` already has the same shape
        # as `shape_ref`, so a `fill`/`Alloc` is unnecessary.

        # XXX FIXME TODO: The only way this can be determined is if one
        # absolutely knows that the shapes of `shape_ref` and `values_ref` are
        # equal.
        # This is an old rewrite, and it's only a
        # "specialization/stabilization", so we're going to leave it be for
        # now.
        return [values_ref]

    if shape_ref.type.broadcastable == out_type.broadcastable:
        # In this case, we assume that some broadcasting is needed (otherwise
        # the condition above would've been true), so we replace the `fill`
        # with an `Alloc`.
        o = broadcast_like(values_ref, shape_ref, fgraph, dtype=values_ref.dtype)
        copy_stack_trace(node.outputs[0], o)
        return [o]

    return


# Register this after stabilize at 1.5 to make sure stabilize don't
# get affected by less canonicalized graph due to alloc.
compile.optdb.register(
    "local_fill_to_alloc", in2out(local_fill_to_alloc), "fast_run", position=1.51
)
# Needed to clean some extra alloc added by local_fill_to_alloc
compile.optdb.register(
    "local_elemwise_alloc", in2out(local_elemwise_alloc), "fast_run", position=1.52
)


@register_canonicalize("fast_compile")
@register_useless
@local_optimizer([fill])
def local_useless_fill(fgraph, node):
    """fill(s,v) -> v

    This optimization is only needed in FAST_COMPILE to make the code
    more readable. Normally, it is done by the local_fill_to_alloc
    opt.

    """
    r, v = node.inputs
    out_type = node.outputs[0].type

    if (
        v.type.dtype == out_type.dtype
        and v.type.broadcastable == out_type.broadcastable
    ):
        return [v]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@local_optimizer([Alloc])
def local_useless_alloc(fgraph, node):
    """
    If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.
    """
    if not isinstance(node.op, Alloc):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]

    if (
        inp.type.dtype == output.type.dtype
        and inp.type.broadcastable == output.type.broadcastable
    ):
        if inp.ndim == 0:
            return [inp]
        else:
            return [
                Assert("Shapes must be equal")(
                    inp, at_all(eq(inp.shape, node.inputs[1:]))
                )
            ]


@register_specialize
@register_stabilize
@register_canonicalize
@local_optimizer([Alloc])
def local_alloc_sink_dimshuffle(fgraph, node):
    r"""Convert broadcastable leading dimensions in an `Alloc` to `DimShuffle`\s."""
    op = node.op
    if not isinstance(op, Alloc):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]

    # Check if alloc adds a broadcastable dimension with shape 1.
    output_shape = node.inputs[1:]
    num_dims_with_size_1_added_to_left = 0
    for i in range(len(output_shape) - inp.ndim):
        if extract_constant(output_shape[i], only_process_constants=True) == 1:
            num_dims_with_size_1_added_to_left += 1
        else:
            break

    new_output_shape = output_shape[num_dims_with_size_1_added_to_left:]
    if num_dims_with_size_1_added_to_left > 0 and len(new_output_shape) >= inp.ndim:
        if (
            output.broadcastable[num_dims_with_size_1_added_to_left:]
            == inp.broadcastable
        ):
            inner = inp
        else:
            inner = op(*([inp] + new_output_shape))
        dimshuffle_new_order = ["x"] * num_dims_with_size_1_added_to_left + list(
            range(len(new_output_shape))
        )
        return [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]


@local_optimizer([AllocEmpty])
def local_alloc_empty_to_zeros(fgraph, node):
    """This convert AllocEmpty to Alloc of 0.

    This help investigate NaN with NanGuardMode.  Not registered by
    default. To activate it, use the Aesara flag
    optimizer_including=alloc_empty_to_zeros. This also enable
    the GPU version of this optimizations.

    """
    if isinstance(node.op, AllocEmpty):
        return [zeros(node.inputs, dtype=node.outputs[0].dtype)]


compile.optdb.register(
    "local_alloc_empty_to_zeros",
    in2out(local_alloc_empty_to_zeros),
    # After move to gpu and merge2, before inplace.
    "alloc_empty_to_zeros",
    position=49.3,
)


@register_specialize
@register_canonicalize
@local_optimizer([Shape])
def local_shape_to_shape_i(fgraph, node):
    if isinstance(node.op, Shape):
        # This optimization needs ShapeOpt and fgraph.shape_feature
        if not hasattr(fgraph, "shape_feature"):
            return
        shape_feature = fgraph.shape_feature
        ret = shape_feature.make_vector_shape(node.inputs[0])

        # We need to copy over stack trace from input to output
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_specialize
@register_canonicalize
@local_optimizer([Shape_i])
def local_track_shape_i(fgraph, node):
    if not isinstance(node.op, Shape_i):
        return False

    try:
        shape_feature = fgraph.shape_feature
    except AttributeError:
        return False

    if node not in shape_feature.scheduled:
        return False

    # Don't unschedule node as it could be reinserted in the
    # fgraph as we don't change it in the shapefeature internal
    # structure.
    replacement = shape_feature.scheduled[node]
    return [shape_feature.shape_of[replacement][node.op.i]]


@register_useless
@register_canonicalize("fast_compile")
@register_specialize
@local_optimizer([Elemwise])
def local_useless_elemwise(fgraph, node):
    """
    eq(x, x) -> 1
    neq(x, x) -> 0
    mul(x) -> x
    add(x) -> x
    identity(x) -> x
    and(x, 1) -> x  (if x.dtype == 'bool')
    and(x, 0) -> zeros_like(x)
    or(x, 0) -> x
    or(x, 1) -> ones_like(x)  (if x.dtype == 'bool')
    xor(x, x) -> zeros_like(x)

    """
    if isinstance(node.op, Elemwise):
        # We call zeros_like and one_like with opt=True to generate a
        # cleaner graph.
        dtype = node.outputs[0].dtype

        if node.op.scalar_op == aes.eq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be true
                ret = ones_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]
        elif node.op.scalar_op == aes.neq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be false
                ret = zeros_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]

        elif node.op.scalar_op == aes.mul and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]

        elif node.op.scalar_op == aes.add and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]
        elif node.op.scalar_op == aes.identity and len(node.inputs) == 1:
            return [node.inputs[0]]

        elif isinstance(node.op.scalar_op, aes.AND) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[1], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[1].astype(node.outputs[0].dtype)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[0].astype(node.outputs[0].dtype)]

        elif isinstance(node.op.scalar_op, aes.OR) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[1].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [ones_like(node.inputs[1], dtype=dtype, opt=True)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[0].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [ones_like(node.inputs[0], dtype=dtype, opt=True)]

        elif isinstance(node.op.scalar_op, aes.XOR) and len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]


@register_specialize
@local_optimizer([Elemwise])
def local_alloc_unary(fgraph, node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)"""
    if isinstance(node.op, Elemwise) and len(node.inputs) == 1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            # at.alloc does not preserve the stacktrace of v,
            # so we need to copy it over from x.
            copy_stack_trace(node.outputs[0], v)
            ret = alloc(cast(v, node.outputs[0].dtype), *shp)

            # at.cast does not preserve the stacktrace of x,
            # so we need to copy it over to the output.
            copy_stack_trace([node.outputs[0], a], ret)
            return [ret]


@register_canonicalize
@register_specialize
@local_optimizer([Elemwise])
def local_cast_cast(fgraph, node):
    """cast(cast(x, dtype1), dtype2)

    when those contrain:
    dtype1 == dtype2
    OR the base dtype is the same (int, uint, float, complex)
          and the first cast cause an upcast.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(node.op.scalar_op, aes.Cast):
        return
    x = node.inputs[0]
    if (
        not x.owner
        or not isinstance(x.owner.op, Elemwise)
        or not isinstance(x.owner.op.scalar_op, aes.Cast)
    ):
        return

    type1 = x.owner.op.scalar_op.o_type
    type2 = node.op.scalar_op.o_type
    base = x.owner.inputs[0]

    if type1 == type2:
        # We don't need to copy over any stack traces here
        return [x]

    if is_an_upcast(base.dtype, type1.dtype):
        # Checking for further redundancy. Eg: int8 -> int32 -> int8
        if type2.dtype == base.dtype:
            return x.owner.inputs
        else:
            # Apply the second cast only
            v = node.op(base)
            # Copy stack trace from the output of the original cast
            copy_stack_trace(node.outputs[0], v)
            return [v]


def is_an_upcast(type1, type2):
    """Given two data types (as strings), check if converting to
    type2 from type1 constitutes an upcast.
    Differs from aesara.scalar.upcast

    """
    category = {
        # The first number in the pair is the dtype (bool, uint, int, float,
        # complex). Conversion from higher to lower is never an upcast.
        # The second number roughly indicates the precision. Again, conversion
        # from higher to lower is never an upcast.
        "bool": (0, 0),
        "uint8": (1, 1),
        "uint16": (1, 2),
        "uint32": (1, 3),
        "uint64": (1, 4),
        "int8": (2, 1),
        "int16": (2, 2),
        "int32": (2, 3),
        "int64": (2, 4),
        "float16": (3, 1.5),
        "float32": (3, 2.5),
        "float64": (3, 3.5),
        "complex64": (4, 3),
        "complex128": (4, 4),
    }

    cat1 = category[type1]
    cat2 = category[type2]

    if cat2[0] >= cat1[0] and cat2[1] > cat1[1]:
        return True
    else:
        return False


@register_useless
@register_specialize
@local_optimizer(None)
def local_remove_useless_assert(fgraph, node):
    if not isinstance(node.op, CheckAndRaise):
        return False

    new_conds = []
    n_conds = len(node.inputs[1:])
    for c in node.inputs[1:]:
        try:
            const = get_scalar_constant_value(c)

            if 0 != const.ndim or const == 0:
                # Should we raise an error here? How to be sure it
                # is not caught?
                new_conds.append(c)
        except NotScalarConstantError:
            new_conds.append(c)

    if len(new_conds) == 0:
        return [node.inputs[0]]

    if len(new_conds) < n_conds:
        new_var = node.op(*(node.inputs[:1] + new_conds))
        copy_stack_trace(node.outputs[0], new_var)
        return [new_var]


@local_optimizer([Assert])
def local_remove_all_assert(fgraph, node):
    """An optimization disabled by default that removes all asserts from
    the graph.

    Notes
    -----
    See the :ref:`unsafe` section to know how to enable it.

    """
    if not isinstance(node.op, Assert):
        return

    return [node.inputs[0]]


compile.optdb["canonicalize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["stabilize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["specialize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["useless"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)


@register_canonicalize
@local_optimizer([Elemwise])
def local_upcast_elemwise_constant_inputs(fgraph, node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return
    try:
        shape_i = fgraph.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, Elemwise):
        scalar_op = node.op.scalar_op
        # print "aa", scalar_op.output_types_preference
        if getattr(scalar_op, "output_types_preference", None) in (
            aes.upgrade_to_float,
            aes.upcast_out,
        ):
            # this is the kind of op that we can screw with the input
            # dtypes by upcasting explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        # works only for scalars
                        cval_i = get_scalar_constant_value(
                            i, only_process_constants=True
                        )
                        if all(i.broadcastable):
                            new_inputs.append(
                                shape_padleft(cast(cval_i, output_dtype), i.ndim)
                            )
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(
                                alloc(
                                    cast(cval_i, output_dtype),
                                    *[shape_i(d)(i) for d in range(i.ndim)],
                                )
                            )
                            # print >> sys.stderr, "AAA",
                            # *[Shape_i(d)(i) for d in range(i.ndim)]
                    except NotScalarConstantError:
                        # for the case of a non-scalar
                        if isinstance(i, TensorConstant):
                            new_inputs.append(cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if not node.outputs[0].type.is_super(rval[0].type):
                    # This can happen for example when floatX=float32
                    # and we do the true division between and int64
                    # and a constant that will get typed as int8.

                    # As this is just to allow merging more case, if
                    # the upcast don't work, we can just skip it.
                    return

                # Copy over output stacktrace from before upcasting
                copy_stack_trace(node.outputs[0], rval)
                return rval


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Rebroadcast])
def local_useless_rebroadcast(fgraph, node):
    """Remove `Rebroadcast` if it does not actually change the broadcasting pattern."""
    if isinstance(node.op, Rebroadcast):
        x = node.inputs[0]
        if np.all(x.broadcastable == node.outputs[0].broadcastable):
            # No broadcastable flag was modified
            # No need to copy over stack trace,
            # because x should already have a stack trace.
            return [x]
        else:
            # Keep the flags that modify something
            new_axis = {}
            for dim, bc in list(node.op.axis.items()):
                if x.broadcastable[dim] != bc:
                    new_axis[dim] = bc
            if new_axis == node.op.axis:
                # All flags are useful
                return
            else:
                r = Rebroadcast(*list(new_axis.items()))(x)
                # Copy over stacktrace from previous output
                copy_stack_trace(node.outputs, r)
                return [r]


@register_canonicalize
@register_specialize
@local_optimizer([Rebroadcast])
def local_rebroadcast_lift(fgraph, node):
    """
    Lifts Rebroadcast through unary Elemwise operations,
    and merges consecutive Rebroadcasts.

    Rebroadcast(Elemwise(x)) => Elemwise(Rebroadcast(x))
    Rebroadcast(Rebroadcast(x)) => Rebroadcast(x)

    """
    op = node.op
    if not isinstance(op, Rebroadcast):
        return False

    inp = node.inputs[0]
    inode = inp.owner
    if inode and isinstance(inode.op, Elemwise) and len(inode.inputs) == 1:
        # It may happen that `input` has no client because this optimization
        # is called from `apply_rebroadcast_opt`, which in particular is used
        # by the `unbroadcast` function before we are in the actual function
        # compilation phase.
        if len(fgraph.clients.get(inp, ())) == 1:
            rebroadcasted = Rebroadcast(*list(op.axis.items()))(inode.inputs[0])
            # Copy over stacktrace from previous output (after rebroadcasting)
            # to new output, because an error in the new graph right after
            # rebroadcasting must have been caused by the previous rebroadcasting.
            copy_stack_trace(node.outputs, rebroadcasted)

            rval = inode.op.make_node(rebroadcasted).outputs

            # Copy over stacktrace from previous output (after rebroadcasting)
            # and input (after elemwise operation) to new output, because an
            # error in the new graph could have been caused by either of the
            # two ops.
            copy_stack_trace(node.outputs + node.inputs, rval)

            return rval
    if inode and isinstance(inode.op, Rebroadcast):
        # the "axis" specification in the outer Rebroadcast overrides
        # the axis of the inner one
        axis = inode.op.axis.copy()
        axis.update(op.axis)
        iinput = inode.inputs[0]

        rval = [Rebroadcast(*list(axis.items()))(iinput)]

        # Copy over stacktrace from previous output (after second rebroadcast)
        # and from previous input (after first rebroadcast op) because an error in
        # the new graph could have been caused by either of the two
        # rebroadcast ops.
        copy_stack_trace(node.outputs + node.inputs, rval)
        return rval


def apply_rebroadcast_opt(rval):
    """
    Apply as many times as required the optimization local_useless_rebroadcast
    and local_rebroadcast_lift.

    Parameters
    ----------
    rval: a Variable

    Returns
    -------
    A Variable (the same if no optimization can be applied)

    """

    fg = FunctionGraph([], [])
    changed = True
    while changed and rval.owner:
        changed = False
        rval2 = local_useless_rebroadcast.transform(fg, rval.owner)
        if rval2:
            assert len(rval2) == 1
            rval = rval2[0]
            changed = True
        if rval.owner:
            rval2 = local_rebroadcast_lift.transform(fg, rval.owner)
            if rval2:
                assert len(rval2) == 1
                rval = rval2[0]
                changed = True
    return rval


@register_specialize
@register_canonicalize
@register_useless
@local_optimizer([Join])
def local_join_1(fgraph, node):
    """Join(i, x) => x

    Remove Join() when only one element is joined.

    """
    if not isinstance(node.op, Join):
        return
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        # We don't need to copy over any stacktrace here, because the
        # input variable should already have its own stacktrace.
        return [tensors[0]]


# TODO: merge in local_useless_join
@register_useless
@register_specialize
@register_canonicalize
@local_optimizer([Join])
def local_join_empty(fgraph, node):
    """Join(i, x, y, empty) => Join(i, x, y)

    Remove empty inputs to joins. The empty inputs can be anywhere.

    """
    if not isinstance(node.op, Join):
        return
    new_inputs = []
    try:
        join_idx = get_scalar_constant_value(
            node.inputs[0], only_process_constants=True
        )
    except NotScalarConstantError:
        return
    for idx in range(1, len(node.inputs)):
        inp = node.inputs[idx]
        # We can not use size == 0,, as this can change shape from 3,0
        # to 2,0.  This trigger DebugMode error. This happen with
        # stack(...,[]) as this add a dimshuffle on [], that add a
        # dimensions with shape 1.
        if isinstance(inp, Constant) and inp.data.shape[join_idx] == 0:
            continue
        new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        if len(new_inputs) == 0:
            # at.join do not work in that case.
            # constant folding will take care of this case.
            return
        ret = join(node.inputs[0], *new_inputs)
        o = node.outputs[0]
        if ret.dtype != o.dtype:
            # Join can upcast some inputs
            return

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)

        if not o.type.is_super(ret.type):
            assert ret.dtype == o.dtype
            assert ret.ndim == o.ndim
            ret = patternbroadcast(ret, node.outputs[0].broadcastable)

        # Copy over stacktrace from previous output
        # (after patternbroadcast op) for same reasons as before.
        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_specialize
@register_canonicalize
@register_useless
@local_optimizer([Join])
def local_join_make_vector(fgraph, node):
    r"""Merge `MakeVector` inputs within a `Join`.

    For example:

        Join(0, make_vector1, make_vector2, ...) => Join(0, make_vector12, ...)

    This in combination with the `local_join_1` optimization can make `Join`\s
    completely disappear.
    """
    if not isinstance(node.op, Join) or node.outputs[0].ndim != 1:
        return
    new_inputs = [node.inputs[1]]
    for idx in range(2, len(node.inputs)):
        inp = node.inputs[idx]
        if (
            inp.owner
            and isinstance(inp.owner.op, MakeVector)
            and new_inputs[-1].owner
            and isinstance(new_inputs[-1].owner.op, MakeVector)
            and
            # MakeVector have a dtype parameter
            inp.owner.op == new_inputs[-1].owner.op
        ):
            inps = new_inputs[-1].owner.inputs + inp.owner.inputs
            new_inputs[-1] = inp.owner.op(*inps)

            # Copy over stacktrace from previous output (after join op)
            # to new intermediate output, because an error in the intermediate
            # op must be caused by an error in the old join op.
            copy_stack_trace(node.outputs, new_inputs[-1])
        else:
            new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        ret = join(node.inputs[0], *new_inputs)

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_useless("local_remove_switch_const_cond")
@register_canonicalize("fast_compile", "local_remove_switch_const_cond")
@register_specialize
@local_optimizer([Elemwise])
def local_useless_switch(fgraph, node):
    """
    This optimization makes the following changes in the graph:

    ``at.switch(cond, left, right)`` ->
            ``if cond is constant and cond == 0``: right
            ``if cond is constant and cond != 0``: left
            ``if left is right`` -> ``left``

    and

    ``at.switch(le(shape_i{id}(X), 0), 0, shape_i{id}(X))`` -> ``shape_i{id}(X)``

    """
    if isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, aes.Switch):

        cond = extract_constant(node.inputs[0], only_process_constants=True)

        if (isinstance(cond, np.ndarray) and cond.ndim == 0) or isinstance(
            cond, (np.number, np.bool_)
        ):
            if cond == 0:
                correct_out = node.inputs[2]
            else:
                correct_out = node.inputs[1]

            if correct_out.dtype != node.outputs[0].dtype:
                out = cast(correct_out, node.outputs[0].dtype)
            else:
                out = correct_out

            out_shape = broadcast_shape(*node.inputs)
            out = alloc(out, *out_shape)

            # Copy over stacktrace from selected output to new output
            copy_stack_trace(node.outputs + correct_out, out)
            return [out]

        # if left is right -> left
        if node.inputs[1] is node.inputs[2]:
            # Note: No need to copy over stacktrace, because the input node
            # already has its own stacktrace
            if cond.type.is_super(node.inputs[1].type):
                return [node.inputs[1]]

            ret = fill(cond, node.inputs[1])

            # Copy over stacktrace from switch output and correct branch
            copy_stack_trace(node.outputs + node.inputs[1], ret)
            return [ret]

        # This case happens with scan.
        # Elemwise{switch}(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)
        left = node.inputs[1]
        right = node.inputs[2]
        cond_var = node.inputs[0]
        if (
            cond_var.owner
            and isinstance(cond_var.owner.op, Elemwise)
            and isinstance(cond_var.owner.op.scalar_op, aes.LE)
            and cond_var.owner.inputs[0].owner
            and isinstance(cond_var.owner.inputs[0].owner.op, Shape_i)
            and extract_constant(cond_var.owner.inputs[1], only_process_constants=True)
            == 0
            and extract_constant(left, only_process_constants=True) == 0
            and right is cond_var.owner.inputs[0]
        ):
            assert node.outputs[0].type.is_super(right.type)
            # No need to copy over stacktrace, because the right input node
            # already has its own stacktrace
            return [right]
        return False
    return False


@register_canonicalize
@local_optimizer([Elemwise])
def local_merge_switch_same_cond(fgraph, node):
    """
    Merge add/sub/mul/div/minimum/maximum/... of switches sharing the same
    condition, to enable further simplification of their branches
    Example: switch(c, a, b) + switch(c, x, y) -> switch(c, a+x, b+y)
    """
    # node must be binary elemwise or add or mul
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, (aes.BinaryScalarOp, aes.Add, aes.Mul)
    ):
        return
    # all inputs must be switch
    if not all(
        s.owner
        and isinstance(s.owner.op, Elemwise)
        and isinstance(s.owner.op.scalar_op, aes.Switch)
        for s in node.inputs
    ):
        return
    # all switch conditions must be the same
    cond = node.inputs[0].owner.inputs[0]
    if not all(s.owner.inputs[0] is cond for s in node.inputs[1:]):
        return
    # pull out switch
    return [
        switch(
            cond,
            node.op(*[s.owner.inputs[1] for s in node.inputs]),
            node.op(*[s.owner.inputs[2] for s in node.inputs]),
        )
    ]


@register_useless
@register_canonicalize
@register_stabilize
@local_optimizer([Tile])
def local_useless_tile(fgraph, node):
    """Tile(x, (1,)*N) -> x

    This is useless tile. (1,)*N, just mean a vector with all element
    being 1.

    """
    if isinstance(node.op, Tile):
        try:
            a = get_scalar_constant_value(node.inputs[1], only_process_constants=True)
            if a == 1:
                try:
                    l = get_vector_length(node.inputs[1])
                    if l == node.inputs[0].ndim:
                        # No need to copy over any stacktrace as previous
                        # input variable already has a stacktrace
                        return [node.inputs[0]]
                    elif l < node.inputs[0].ndim:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        return [node.inputs[0]]
                    else:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        x_nd = node.inputs[0].ndim
                        broad = ["x"] * (l - x_nd) + range(x_nd)
                        ret = node.inputs[0].dimshuffle(broad)
                        # Copy over stacktrace from previous output node,
                        # and from node before tiling operation.
                        copy_stack_trace(node.outputs + node.inputs[0], ret)
                        return [ret]
                except ValueError:
                    return
        except NotScalarConstantError:
            return


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Split])
def local_useless_split(fgraph, node):
    """Split{n_splits=1}(x, y) -> x

    Remove Split with only 1 split.

    """
    if isinstance(node.op, Split):
        if node.op.len_splits == 1:
            x, axis, splits = node.inputs
            out = assert_op(x, eq(splits.shape[0], 1))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(node.outputs, out)
            out2 = assert_op(out, eq(x.shape[axis], splits[0]))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(out, out2)

            return [out2]


@register_canonicalize
@register_stabilize
@local_optimizer([Flatten])
def local_flatten_lift(fgraph, node):
    """
    Flatten(UnaryElemwise(x)) -> UnaryElemwise(Flatten(x))

    This optimization is needed by optimization
    log1msigm_to_softplus to get applied when there is a flatten.

    """
    if (
        isinstance(node.op, Flatten)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        f = node.op(node.inputs[0].owner.inputs[0])

        # Copy over stacktrace from previous output node (flatten op),
        # since this is the op which may cause an error for f.
        copy_stack_trace(node.outputs, f)

        e = node.inputs[0].owner.op(f)

        # Copy over stacktrace from previous output node and from unary
        # elementwise output node since if there was an error, it would
        # probably have come from that operation.
        copy_stack_trace(node.outputs + [node.inputs[0]], e)

        return [e]


def local_reshape_chain(op):
    @local_optimizer([op])
    def f(fgraph, node):
        """
        Reshape(Reshape(shape1),shape2) -> Reshape(shape2)

        """
        if not check_chain(node, op, op):
            return False

        # TODO: this can permit a failing program to run by eliminating
        #       the lower reshape
        rval = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])

        # Copy over stacktrace from previous output node, as any error
        # in new computational graph would have been caused by last op
        # in the old computational graph.
        copy_stack_trace(node.outputs, rval)

        # It might happen that the desired output of this node has a
        # broadcastable pattern that does not match that of 'rval'. This is
        # when originally, we were able to figure out that one of the
        # dimensions of the reshape is one, but some other transformation
        # replaced the shape by one for which this cannot be guessed.
        # We should try to figure out why we lost the information about this
        # constant value... but in the meantime, better not apply this
        # optimization.
        if rval.broadcastable == node.outputs[0].broadcastable:
            return [rval]
        else:
            return False

    return f


register_canonicalize(local_reshape_chain(Reshape), name="local_reshape_chain")


@register_useless
@register_canonicalize
@register_stabilize
@local_optimizer([Reshape])
def local_useless_reshape(fgraph, node):
    """
    Remove two kinds of useless reshape.

    Remove Reshape when both the input and output have a single dimension.
    Remove Reshape when reshaping to the shape of the input.

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    if inp.ndim != output.ndim:
        return False

    # Simple case: both input and output have a single dimension.
    # This could hide errors if the user provides inconsistent shapes.
    if inp.ndim == 1 and output.ndim == 1 and inp.broadcastable == output.broadcastable:
        return [inp]

    # Second case: all the shapes match the input shape
    # Match Reshape(x, x.shape)
    if output_shape.owner and isinstance(output_shape.owner.op, Shape):
        shape_input = output_shape.owner.inputs[0]
        if shape_input == inp:
            return [inp]

    # Match Reshape(x, [x.shape[0], ..., x.shape[-1]]), accounting for
    # broadcastable and constant dimensions
    if output_shape.owner and isinstance(output_shape.owner.op, MakeVector):
        output_shape_is = output_shape.owner.inputs

        shape_feature = getattr(fgraph, "shape_feature", None)

        nb_m1 = 0
        shape_match = [False] * inp.ndim
        for dim in range(inp.ndim):
            outshp_i = output_shape_is[dim]
            # Match Shape_i{dim}(input)
            if (
                outshp_i.owner
                and isinstance(outshp_i.owner.op, Shape_i)
                and outshp_i.owner.op.i == dim
                and outshp_i.owner.inputs[0] == inp
            ):
                shape_match[dim] = True
                continue

            # Match Shape(input)[dim]
            if (
                outshp_i.owner
                and isinstance(outshp_i.owner.op, Subtensor)
                and len(outshp_i.owner.inputs) == 2
                and extract_constant(outshp_i.owner.inputs[1]) == dim
            ):
                subtensor_inp = outshp_i.owner.inputs[0]
                if subtensor_inp.owner and isinstance(subtensor_inp.owner.op, Shape):
                    shape_input_i = subtensor_inp.owner.inputs[0]
                    if shape_input_i == inp:
                        shape_match[dim] = True
                        continue

            # Match 1 if input.broadcastable[dim] is True
            cst_outshp_i = extract_constant(outshp_i, only_process_constants=1)
            if inp.broadcastable[dim] and cst_outshp_i == 1:
                shape_match[dim] = True
                continue

            # Match -1
            if cst_outshp_i == -1:
                shape_match[dim] = True
                nb_m1 += 1
                continue

            # Match shape_of[input][dim] or its constant equivalent
            if shape_feature:
                inpshp_i = shape_feature.get_shape(inp, dim)
                if inpshp_i == outshp_i or (
                    extract_constant(inpshp_i, only_process_constants=1)
                    == extract_constant(outshp_i, only_process_constants=1)
                ):
                    shape_match[dim] = True
                    continue

        if all(shape_match) and nb_m1 <= 1:
            return [inp]

        # TODO later: if all the shapes except one match, we may want to
        # consider it useless as well, like we do in the 1-dim case.
        return False


@register_canonicalize
@local_optimizer([Reshape])
def local_reshape_to_dimshuffle(fgraph, node):
    """
    Broadcastable dimensions in Reshape are replaced with dimshuffle.

    The goal is to avoid using reshape to add or remove broadcastable
    dimensions, but use dimshuffle instead, so dimshuffles can cancel out
    or be removed later on.

    For example:
        - reshape(x, (1, n)) --> dimshuffle{x,0}(reshape(x, (n,))
        - reshape(x, (1, m, 1, n, 1, 1))
          --> dimshuffle{x,0,x,1,x,x}(reshape(x, (m, n)))
    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    inp = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    dimshuffle_new_order = []
    new_output_shape = []
    index = 0  # index over the output of the new reshape
    for i in range(output.ndim):
        # Since output_shape is a symbolic vector, we trust extract_constant
        # to go through however it is formed to see if its i-th element is 1.
        # We need only_process_constants=False for that.
        dim = extract_constant(
            output_shape[i], only_process_constants=False, elemwise=False
        )
        if dim == 1:
            dimshuffle_new_order.append("x")
        else:
            dimshuffle_new_order.append(index)
            new_output_shape.append(dim)
            index = index + 1
    if index != output.ndim:
        inner = op.__class__(len(new_output_shape))(inp, new_output_shape)
        copy_stack_trace(output, inner)
        new_node = [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]
        copy_stack_trace(output, new_node)
        return new_node


@register_canonicalize
@register_stabilize
@local_optimizer([Reshape])
def local_reshape_lift(fgraph, node):
    """
    Reshape(UnaryElemwise(x)) -> UnaryElemwise(Reshape(x))

    This optimization is needed by optimization
    log1msigm_to_softplus to get applied when there is a reshape.

    """
    if (
        isinstance(node.op, Reshape)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        r = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
        # Copy stacktrace from previous Reshape op, as an error in new
        # Reshape op could only have been caused by old one.
        copy_stack_trace(node.outputs, r)

        e = node.inputs[0].owner.op(r)
        # Copy stacktrace from both previous Reshape and UnaryElemwise op
        # because an error in new cg could have been caused by either ops.
        copy_stack_trace(node.outputs + node.inputs, e)

        # In rare case the original broadcast was (False, True), but
        # the new one is (False, False). So don't crash in that case.
        if not node.outputs[0].type.is_super(e.type):
            re = patternbroadcast(e, node.outputs[0].broadcastable)

            # Copy over stack trace.
            # If the graph fails it is usually due to the fact that a dimension
            # that should be broadcastable does not actually have length 1,
            copy_stack_trace(e, re)
        else:
            re = e

        return [re]


register_canonicalize(OpRemove(tensor_copy), name="remove_tensor_copy")


@local_optimizer(None)
def constant_folding(fgraph, node):

    if not node.op.do_constant_folding(fgraph, node):
        return False

    if not all(isinstance(inp, Constant) for inp in node.inputs):
        return False

    storage_map = {i: [i.data] for i in node.inputs}
    compute_map = {i: [True] for i in node.inputs}
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling=[])
    required = thunk()

    # A node whose inputs are all provided should always return successfully
    assert not required

    rval = []
    for output in node.outputs:
        data = storage_map[output][0]
        assert compute_map[output][0], (output, data)

        # TODO: `Type` itself should provide an interface for constructing
        # instances appropriate for a given constant.
        # TODO: Add handling for sparse types.
        if isinstance(output.type, DenseTensorType):
            output_type = TensorType(
                output.type.dtype,
                tuple(s == 1 for s in data.shape),
                name=output.type.name,
            )
        else:
            output_type = output.type

        v = output_type.make_constant(data)

        # We need to "narrow" types when we have additional information,
        # and not "broaden" them.  This is a case in which types are
        # unnecessarily "broadened"
        # assert not hasattr(output.type, "broadcastable") or output.type.broadcastable == tuple(s == 1 for s in data.shape)

        copy_stack_trace(output, v)

        rval.append(v)

    return rval


topo_constant_folding = in2out(
    constant_folding, ignore_newtrees=True, name="topo_constant_folding"
)
register_canonicalize(topo_constant_folding, "fast_compile", final_opt=True)
register_uncanonicalize(topo_constant_folding, "fast_compile", final_opt=True)
register_stabilize(topo_constant_folding, "fast_compile", final_opt=True)
register_specialize(topo_constant_folding, "fast_compile", final_opt=True)


def local_elemwise_fusion_op(op_class, max_input_fct=lambda node: 32, maker=None):
    r"""Create a recursive function that fuses `Elemwise` `Op`\s.

    The basic idea is that we loop through an `Elemwise` node's inputs, find
    other `Elemwise` nodes, determine the scalars input types for all of the
    `Elemwise` `Op`\s, construct a new scalar `Op` using the scalar input types
    and each `Elemwise`'s scalar `Op`, and use the composite scalar `Op` in a
    new "fused" `Elemwise`.

    It's parameterized in order to work for `Elemwise` and `GpuElemwise` `Op`\s.

    Parameters
    ----------
    op_class : type
        `GpuElemwise` or `Elemwise` class (the one that we want to fuse)
    max_input_fct : callable
        A function that returns the maximum number of inputs that this `Elemwise`
        can take (useful for `GpuElemwise`).  The GPU kernel currently has a
        limit of 256 bytes for the size of all parameters passed to it. As
        currently we pass a lot of information only by parameter, we must limit how
        many `Op`\s we fuse together to avoid busting that 256 limit.

        On the CPU we limit to 32 input variables since that is the maximum
        NumPy support.

    maker: callable
        A function with the signature ``(node, *args)`` that constructs an
        `op_class` instance (e.g. ``op_class(*args)``).

    """
    if maker is None:

        def maker(node, scalar_op):
            return op_class(scalar_op)

    def local_fuse(fgraph, node):
        r"""Fuse `Elemwise` `Op`\s in a node.

        As part of specialization, we fuse two consecutive `Elemwise` `Op`\s of the
        same shape.

        For mixed dtype, we let the `Composite` `Op` do the cast. It lets the C
        compiler do the cast.

        The number of dimensions is validated at call time by Aesara itself.

        """
        # META TODO:  PUT THESE THINGS IN TRAC, NOT TODO NOTES!!
        # TODO: use broadcast flag?

        # TODO: don't do this optimization as a localOptimizer.
        # Analyze the graph in terms of elemwise subgraphs, and then
        # replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't
        # fit within the parameter space of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs
        # have multiple clients. This can't be done with a local
        # optimiser.

        # TODO: Related: Support composites with multiple outputs

        # TODO: Use Composite to combine Elemwise and Reduce
        # operations.  We have to loop over the data anyway... might
        # as well sum it up while we're at it (this can be trickier
        # than i'm making it seound here. The data-traversal should be
        # done contiguously, and the summing-up might not be easy or
        # worthwhile if the summation axis doesn't line up with a
        # contiguous dimension)

        if type(node.op) is not op_class:
            return False

        if len(node.outputs) > 1:
            # We don't support fusion for nodes with multiple outputs.
            return

        inputs = []  # inputs of the new Elemwise op.
        s_inputs = []  # inputs of the new scalar op used by the Composite.
        # Inputs of the new scalar op that represents the current node.
        s_g = []

        # There is a hard limit of 256 bytes for the formal argument list to a
        # GPU kernel function.
        max_nb_input = max_input_fct(node)
        # The number of inputs to the new fused op if we do not fuse more
        # inputs.
        new_nb_input = len(node.inputs)
        # Did we fuse something?
        # Needed as we can fuse unary op that don't change the number of
        # inputs.
        # And there is a case where the inputs are the same as the current
        # node. That won't change the number of inputs of the new op.
        fused = False

        for i in node.inputs:
            do_fusion = False
            # Will store inputs of the fused node that are not currently inputs
            # of the node we want to create (to avoid duplicating inputs).
            tmp_input = []
            # Same as tmp_input, but for scalars.
            tmp_scalar = []

            # We should not check the number of inputs here
            # As fusing op don't always change the number of input.
            # If a variable is used as multiple into to the same node,
            # we still want to fusion. So we take the set.
            if (
                i.owner
                and isinstance(i.owner.op, op_class)
                and len({n for n, idx in fgraph.clients[i]}) == 1
                and
                # Do not merge elemwise that don't have the same
                # broadcastable pattern to don't redo duplicate
                # computation due to broadcast.
                i.owner.outputs[0].broadcastable == node.outputs[0].broadcastable
            ):
                try:
                    tmp_s_input = []
                    # we should not put duplicate input into s_inputs and inputs
                    for ii in i.owner.inputs:
                        if ii in inputs:
                            tmp_s_input.append(s_inputs[inputs.index(ii)])
                        elif ii in tmp_input:
                            tmp_s_input.append(tmp_scalar[tmp_input.index(ii)])
                        else:
                            tmp = aes.get_scalar_type(ii.type.dtype).make_variable()
                            try:
                                tv = get_test_value(ii)
                                if tv.size > 0:
                                    tmp.tag.test_value = tv.flatten()[0]
                                else:
                                    _logger.warning(
                                        "Cannot construct a scalar test value"
                                        " from a test value with no size: {}".format(ii)
                                    )
                            except TestValueError:
                                pass

                            tmp_s_input.append(tmp)
                            tmp_input.append(ii)
                            tmp_scalar.append(tmp_s_input[-1])

                    s_op = i.owner.op.scalar_op(*tmp_s_input, return_list=True)

                    # If the scalar_op doesn't have a C implementation, we skip
                    # its fusion to allow fusion of the other ops
                    i.owner.op.scalar_op.c_code(
                        s_op[0].owner,
                        "test_presence_of_c_code",
                        ["x" for x in i.owner.inputs],
                        ["z" for z in i.owner.outputs],
                        {"fail": "%(fail)s"},
                    )

                    do_fusion = True

                except (NotImplementedError, MethodNotDefined):
                    _logger.warning(
                        (
                            "Optimization Warning: "
                            f"The Op {i.owner.op.scalar_op} does not provide a C implementation."
                            " As well as being potentially slow, this also disables "
                            "loop fusion."
                        )
                    )
                    do_fusion = False

            # Compute the number of inputs in case we fuse this input.
            # We subtract 1 because we replace the existing input with the new
            # inputs from `tmp_input`.
            new_nb_input_ = new_nb_input + len(tmp_input) - 1

            # If the new input is already an input of the current node, it was
            # already counted when `new_nb_input` was initialized to
            # len(node.inputs).
            # This can happen when a variable is used both by the Elemwise to
            # fuse and the current node.
            for x in tmp_input:
                if x in node.inputs:
                    new_nb_input_ -= 1

            if do_fusion and (new_nb_input_ <= max_nb_input):
                fused = True
                new_nb_input = new_nb_input_
                inputs.extend(tmp_input)
                s_inputs.extend(tmp_scalar)
                s_g.extend(s_op)
            else:
                # We must support the case where the same variable appears many
                # times within the inputs
                if inputs.count(i) == node.inputs.count(i):
                    s = s_inputs[inputs.index(i)]
                else:
                    s = aes.get_scalar_type(i.type.dtype).make_variable()
                    try:
                        if config.compute_test_value != "off":
                            v = get_test_value(i)
                            if v.size > 0:
                                s.tag.test_value = v.flatten()[0]
                    except TestValueError:
                        pass

                    inputs.append(i)
                    s_inputs.append(s)
                s_g.append(s)

        if not fused:
            return False

        if new_nb_input != len(inputs) or len(s_inputs) != len(inputs):
            raise Exception(
                """Something has gone wrong with the elemwise
fusion optimization. We skip this optimization. You can ignore this message,
your code will run correctly, but may be slower."""
            )

        s_new_out = node.op.scalar_op(*s_g, return_list=True)
        try:
            s_new_out[0].owner.op.c_code(
                s_new_out[0].owner,
                "test_presence_of_c_code",
                ["x" for x in s_g],
                ["z" for x in s_new_out],
                {"fail": "%(fail)s"},
            )
        except (NotImplementedError, MethodNotDefined):
            name = str(s_new_out[0].owner.op)
            _logger.warning(
                (
                    "Optimization Warning: "
                    f"The Op {name} does not provide a C implementation."
                    " As well as being potentially slow, this also disables "
                    "loop fusion."
                )
            )
            return False

        # create the composite op.
        composite_op = aes.Composite(s_inputs, s_new_out)

        # create the new node.
        # Do not call make_node to have test_value
        new_node = maker(node, composite_op)(*inputs).owner

        assert len(new_node.outputs) == 1
        assert node.outputs[0].type.dtype == new_node.outputs[0].type.dtype

        if len(new_node.inputs) > max_nb_input:
            _logger.warning(
                "loop fusion failed because Op would exceed" " kernel argument limit."
            )
            return False

        # we fuse as many that we can at the same time to make debug mode faster
        # debug mode will be faster as it won't test all intermediate step.
        while True:
            ret = local_fuse(fgraph, new_node)
            if ret is not False and ret is not None:
                assert len(ret) == len(new_node.outputs)
                assert len(ret) == 1
                new_node = ret[0].owner
            else:
                break

        return new_node.outputs

    return local_fuse


def elemwise_max_input_fct(node):
    # `Elemwise.perform` uses NumPy ufuncs and they are limited to 31 inputs.
    if not config.cxx:
        return 31
    return 1024


local_elemwise_fusion = local_elemwise_fusion_op(Elemwise, elemwise_max_input_fct)


class FusionOptimizer(GlobalOptimizer):
    """Graph optimizer that simply runs local fusion operations.

    TODO: This is basically a `EquilibriumOptimizer`; we should just use that.

    """

    def __init__(self, local_optimizer):
        super().__init__()
        self.optimizer = local_optimizer

    def add_requirements(self, fgraph):
        fgraph.attach_feature(features.ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        nb_iter = 0
        nb_replacement = 0
        nb_inconsistency_replace = 0
        time_toposort = 0
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time
        while did_something:
            t0 = time.time()
            nodelist = list(fgraph.toposort())
            time_toposort += time.time() - t0
            nodelist.reverse()
            did_something = False
            for node in nodelist:
                # Don't try to fuse node that have already been fused.
                if node in fgraph.apply_nodes:
                    new_outputs = self.optimizer(fgraph, node)
                    if new_outputs:
                        assert len(new_outputs) == len(node.outputs)
                        try:
                            fgraph.replace_all_validate(
                                list(zip(node.outputs, new_outputs)),
                                reason=self.__class__.__name__,
                            )
                            did_something = True
                            nb_replacement += 1
                        except InconsistencyError:
                            nb_inconsistency_replace += 1
            nb_iter += 1

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}
        return (
            self,
            nb_iter,
            nb_replacement,
            nb_inconsistency_replace,
            validate_time,
            callback_time,
            callbacks_time,
            time_toposort,
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)
        print(blanc, " time_toposort", prof[7], file=stream)


if config.tensor__local_elemwise_fusion:
    _logger.debug("Enabling Elemwise fusion optimizations in fast_run")
    # Must be after gpu(48.5) and before AddDestroyHandler(49.5)
    fuse_seqopt = SequenceDB()
    fuse_seqopt.register(
        "composite_elemwise_fusion",
        FusionOptimizer(local_elemwise_fusion),
        "fast_run",
        "fusion",
        position=1,
    )
    compile.optdb.register(
        "elemwise_fusion",
        fuse_seqopt,
        "fast_run",
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
        position=49,
    )
else:
    _logger.debug("not enabling optimization fusion elemwise in fast_run")
    compile.optdb.register(
        "elemwise_fusion",
        FusionOptimizer(local_elemwise_fusion),
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
        position=49,
    )


@register_canonicalize
@local_optimizer([Elemwise])
def local_useless_composite(fgraph, node):
    """For elemwise Composite that have multiple outputs, remove the
    outputs that are not used.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, aes.Composite
    ):
        return
    comp = node.op.scalar_op
    idx = [i for i, o_extern in enumerate(node.outputs) if fgraph.clients[o_extern]]
    if len(idx) < len(node.outputs):
        new_outputs = [comp.outputs[i] for i in idx]
        c = aes.Composite(inputs=comp.inputs, outputs=new_outputs)
        e = Elemwise(scalar_op=c)(*node.inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in idx], e))


@register_canonicalize("fast_compile")
@register_useless("fast_compile")
@local_optimizer(None)
def local_view_op(fgraph, node):
    if isinstance(node.op, ViewOp):
        return node.inputs


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([Alloc])
def local_merge_alloc(fgraph, node):
    # This opt takes care of several cases:
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) -> Alloc(m, x, assert(y1, y1==y2), z, w)
    if not isinstance(node.op, Alloc):
        return False
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Alloc):
        return False
    inputs_outer = node.inputs
    inputs_inner = node.inputs[0].owner.inputs
    dims_outer = inputs_outer[1:]
    dims_inner = inputs_inner[1:]
    dims_outer_rev = dims_outer[::-1]
    dims_inner_rev = dims_inner[::-1]
    # check if the pattern of broadcasting is matched, in the reversed ordering.
    # The reverse ordering is needed when an Alloc add an implicit new
    # broadcasted dimensions to its inputs[0]. Eg:
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    i = 0
    for dim_inner, dim_outer in zip(dims_inner_rev, dims_outer_rev):
        if dim_inner != dim_outer:
            if isinstance(dim_inner, Constant) and dim_inner.data == 1:
                pass
            else:
                dims_outer[-1 - i] = Assert(
                    "You have a shape error in your graph. To see a better"
                    " error message and a stack trace of where in your code"
                    " the error is created, use the Aesara flags"
                    " optimizer=None or optimizer=fast_compile."
                )(dim_outer, eq(dim_outer, dim_inner))
        i += 1
    return [alloc(inputs_inner[0], *dims_outer)]


@register_useless("fast_compile")
@local_optimizer([TopKOp])
def local_useless_topk(fgraph, node):
    """
    TopKOp generates two outputs by default
    This opt removes the useless ones

    """
    op = node.op
    if not isinstance(op, TopKOp):
        return
    if not (op.return_values and op.return_indices):
        return False

    x, k = node.inputs
    ret_val = bool(fgraph.clients[node.outputs[0]])
    ret_idx = bool(fgraph.clients[node.outputs[1]])

    if not (ret_val ^ ret_idx):
        # both true -> nothing to remove
        # both false -> let pruner handle
        return False

    old_output = node.outputs[ret_idx]
    new_output = TopKOp(
        axis=op.axis,
        sorted=op.sorted,
        idx_dtype=op.idx_dtype,
        return_values=ret_val,
        return_indices=ret_idx,
    )(x, k)
    copy_stack_trace(node.outputs[0], new_output)
    return {old_output: new_output}


@register_useless
@register_canonicalize
@local_optimizer([SpecifyShape])
def local_useless_SpecifyShape(fgraph, node):
    """Replace ``specify_shape(specify_shape(x, s1), s2)`` with ``specify_shape(x, s1)``."""

    if not isinstance(node.op, SpecifyShape):
        return False

    obj = node.inputs[0]
    if not (obj.owner and isinstance(obj.owner.op, SpecifyShape)):
        return False

    # TODO: We could make sure that the shapes of the two `SpecifyShape`s are
    # the same.

    return [obj]


@register_useless
@register_canonicalize
@local_optimizer([Shape])
def local_Shape_of_SpecifyShape(fgraph, node):
    """Replace ``specify_shape(x, s).shape`` with ``s``."""

    if not isinstance(node.op, Shape):
        return False

    specified_shape = node.inputs[0]

    if not isinstance(getattr(specified_shape.owner, "op", None), SpecifyShape):
        return False

    return [specified_shape.owner.inputs[1].astype(np.int64)]


@register_useless
@register_canonicalize
@local_optimizer([Shape_i])
def local_Shape_i_of_broadcastable(fgraph, node):
    """Replace ``shape_i(x, i)`` with ``1`` when ``x.broadcastable[i]`` is ``True``."""

    if not isinstance(node.op, Shape_i):
        return False

    shape_arg = node.inputs[0]

    if not isinstance(shape_arg.type, TensorType):
        return False

    if shape_arg.broadcastable[node.op.i]:
        return [as_tensor_variable(1, dtype=np.int64)]


@register_useless
@register_canonicalize
@local_optimizer([Unique])
def local_Unique_scalar(fgraph, node):
    """Convert ``unique(x)`` to ``x`` when ``x`` is a scalar."""
    if not isinstance(node.op, Unique):
        return False

    if node.op.return_index or node.op.return_inverse or node.op.return_counts:
        return False

    uniqued_var = node.inputs[0]

    if uniqued_var.ndim != 0:
        return False

    old_out = node.outputs[0]
    res = as_tensor_variable(uniqued_var, ndim=old_out.ndim, dtype=old_out.dtype)
    return [res]


@register_useless
@register_canonicalize
@local_optimizer([Unique])
def local_Unique_Alloc_lift(fgraph, node):
    """Convert ``unique(alloc(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    alloc_var = node.inputs[0]

    if not (alloc_var.owner and isinstance(alloc_var.owner.op, Alloc)):
        return False

    alloced_var, *alloc_shape = alloc_var.owner.inputs

    new_unique, *_ = node.op.make_node(alloced_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@local_optimizer([Unique])
def local_Unique_BroadcastTo_lift(fgraph, node):
    """Convert ``unique(broadcast_to(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    bcast_var = node.inputs[0]

    if not (bcast_var.owner and isinstance(bcast_var.owner.op, BroadcastTo)):
        return False

    bcasted_var, *bcast_shape = bcast_var.owner.inputs

    new_unique, *_ = node.op.make_node(bcasted_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@local_optimizer([Unique])
def local_Unique_Repeat_lift(fgraph, node):
    """Convert ``unique(repeat(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    repeat_var = node.inputs[0]

    if not (repeat_var.owner and isinstance(repeat_var.owner.op, Repeat)):
        return False

    repeated_var, *repeat_shape = repeat_var.owner.inputs

    new_unique, *_ = node.op.make_node(repeated_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@local_optimizer([Unique])
def local_Unique_second(fgraph, node):
    """Convert ``unique(second(x, ...), axis=None)`` to ``second(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    second_var = node.inputs[0]

    if not (
        second_var.owner
        and isinstance(second_var.owner.op, Elemwise)
        and isinstance(second_var.owner.op.scalar_op, aes.Second)
    ):
        return False

    shape_var, seconded_var = second_var.owner.inputs

    new_unique, *_ = node.op.make_node(seconded_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@local_optimizer([BroadcastTo])
def local_remove_scalar_BroadcastTo(fgraph, node):

    bcast_shape = node.inputs[1:]

    if not bcast_shape:
        bcasted_var = node.inputs[0]
        # If this isn't true, the graph is invalid
        assert bcasted_var.ndim == 0
        return [bcasted_var]

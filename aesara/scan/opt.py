"""This module provides optimizations for the `Scan` `Op`."""

import copy
import dataclasses
from itertools import chain
from sys import maxsize
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

import aesara
from aesara import scalar as aes
from aesara import tensor as at
from aesara.compile import optdb
from aesara.compile.function.types import deep_copy_op
from aesara.configdefaults import config
from aesara.graph.basic import (
    Apply,
    Constant,
    Variable,
    clone_replace,
    equal_computations,
    graph_inputs,
    io_toposort,
    is_in_ancestors,
)
from aesara.graph.destroyhandler import DestroyHandler
from aesara.graph.features import ReplaceValidate
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import compute_test_value
from aesara.graph.opt import GlobalOptimizer, in2out, local_optimizer
from aesara.graph.optdb import EquilibriumDB, SequenceDB
from aesara.graph.type import HasShape
from aesara.graph.utils import InconsistencyError
from aesara.scan.op import Scan, ScanInfo
from aesara.scan.utils import (
    ScanArgs,
    compress_outs,
    expand_empty,
    reconstruct_graph,
    safe_new,
    scan_can_remove_outs,
)
from aesara.tensor import basic_opt, math_opt
from aesara.tensor.basic import Alloc, AllocEmpty, get_scalar_constant_value
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import Dot, dot, maximum, minimum
from aesara.tensor.shape import shape
from aesara.tensor.subtensor import (
    IncSubtensor,
    Subtensor,
    get_canonical_form_slice,
    get_idx_list,
    get_slice_elements,
    set_subtensor,
)
from aesara.tensor.var import TensorConstant, get_unique_value


list_opt_slice = [
    math_opt.local_abs_merge,
    math_opt.local_mul_switch_sink,
    basic_opt.local_upcast_elemwise_constant_inputs,
    basic_opt.local_useless_switch,
    basic_opt.constant_folding,
]


@local_optimizer([Scan])
def remove_constants_and_unused_inputs_scan(fgraph, node):
    """Move constants into the inner graph, and remove unused inputs.

    Constants that are in the outer graph are represented by a free symbolic
    variable in the inner graph. If we move them into the inner graph,
    constant-folding can happen in the inner graph.
    This is applied only on sequences and non-sequences,
    not on initial states.

    """
    if not isinstance(node.op, Scan):
        return False
    op = node.op
    op_info = op.info
    # We only need to take care of sequences and other arguments
    st = op_info.n_seqs
    st += int(
        sum(len(x) for x in chain(op_info.mit_mot_in_slices, op_info.mit_sot_in_slices))
    )
    st += op_info.n_sit_sot
    st += op_info.n_shared_outs

    op_ins = op.inner_inputs
    op_outs = op.inner_outputs

    # Corresponds to the initial states, which should stay untouched.
    # We put those variables aside, and put them back at the end.
    out_stuff_inner = op_ins[op_info.n_seqs : st]

    non_seqs = op_ins[st:]
    st = (
        op_info.n_seqs
        + op_info.n_mit_mot
        + op_info.n_mit_sot
        + op_info.n_sit_sot
        + op_info.n_nit_sot
        + op_info.n_shared_outs
        + 1
    )
    outer_non_seqs = node.inputs[st:]
    out_stuff_outer = node.inputs[1 + op_info.n_seqs : st]

    # To replace constants in the outer graph by clones in the inner graph
    givens = {}
    # All the inputs of the inner graph of the new scan
    nw_inner = []
    # Same for the outer graph, initialized w/ number of steps
    nw_outer = [node.inputs[0]]

    all_ins = list(graph_inputs(op_outs))
    for idx in range(op_info.n_seqs):
        node_inp = node.inputs[idx + 1]
        if (
            isinstance(node_inp, TensorConstant)
            and get_unique_value(node_inp) is not None
        ):
            try:
                # This works if input is a constant that has all entries
                # equal
                givens[op_ins[idx]] = node_inp[0]
            except TypeError:
                pass
        elif op_ins[idx] in all_ins:
            # Check for identical other sequence
            identical_seqs = [
                x for x in nw_outer if equal_computations([x], [node_inp])
            ]
            if identical_seqs:
                index = node.inputs.index(identical_seqs[0]) - 1
                givens[op_ins[idx]] = op_ins[index]
            else:
                nw_inner.append(op_ins[idx])
                nw_outer.append(node_inp)

    nw_n_seqs = len(nw_inner)
    # Add outputs stuff
    nw_inner += out_stuff_inner
    nw_outer += out_stuff_outer

    # Look through non sequences
    nw_inner_nonseq = []
    nw_outer_nonseq = []
    for idx, (nw_in, nw_out) in enumerate(zip(non_seqs, outer_non_seqs)):
        if isinstance(nw_out, Constant):
            givens[nw_in] = nw_out
        elif nw_in in all_ins:
            # Indices of elements of nw_outer_nonseq that are equivalent
            # to nw_out.
            identical_nonseq_idx = [
                i
                for (i, x) in enumerate(nw_outer_nonseq)
                if equal_computations([x], [nw_out])
            ]
            if identical_nonseq_idx:
                givens[nw_in] = nw_inner_nonseq[identical_nonseq_idx[0]]
            else:
                nw_inner_nonseq.append(nw_in)
                nw_outer_nonseq.append(nw_out)

    nw_inner.extend(nw_inner_nonseq)
    nw_outer.extend(nw_outer_nonseq)

    if len(nw_inner) != len(op_ins):
        op_outs = clone_replace(op_outs, replace=givens)
        nw_info = dataclasses.replace(
            op_info, n_seqs=nw_n_seqs, n_non_seqs=len(nw_inner_nonseq)
        )
        nwScan = Scan(
            nw_inner,
            op_outs,
            nw_info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )
        nw_outs = nwScan(*nw_outer, return_list=True)
        return dict([("remove", [node])] + list(zip(node.outputs, nw_outs)))
    else:
        return False


@local_optimizer([Scan])
def push_out_non_seq_scan(fgraph, node):
    r"""Push out the variables inside the `Scan` that depend only on non-sequences.

    This optimizations pushes, out of `Scan`'s inner function and into the outer
    function, computation that depends only on non-sequence inputs. Such
    computation ends up being done every iteration on the same values so moving
    it to the outer function to be executed only once, before the `Scan` `Op`,
    reduces the amount of computation that needs to be performed.
    """
    if not isinstance(node.op, Scan):
        return False

    node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs

    local_fgraph_topo = io_toposort(node_inputs, node_outputs)
    local_fgraph_outs_set = set(node_outputs)
    local_fgraph_outs_map = {v: k for k, v in enumerate(node_outputs)}

    to_remove_set = set()
    to_replace_set = set()
    to_replace_map = {}

    def add_to_replace(y):
        to_replace_set.add(y)
        to_replace_map[y] = add_to_replace.n
        add_to_replace.n += 1

    add_to_replace.n = 0

    # The variables that will replace the variables pushed-out of the
    # inner-graph
    replace_with_in = []
    # The variables that have been pushed-out of the graph
    replace_with_out = []

    op = node.op
    # Construct the list of non_sequences to simplify a few things
    inner_non_seqs = op.inner_non_seqs(node_inputs)
    inner_non_seqs_set = set(inner_non_seqs)
    inner_non_seqs_map = {v: k for k, v in enumerate(inner_non_seqs)}

    outer_non_seqs = op.outer_non_seqs(node.inputs)

    inner_seqs = op.inner_seqs(node_inputs)
    outer_seqs = op.outer_seqs(node.inputs)

    assert len(inner_non_seqs) == len(outer_non_seqs)
    assert len(inner_seqs) == len(outer_seqs)

    for nd in local_fgraph_topo:
        if (  # we haven't already looked at this node
            nd not in to_remove_set
            and all(
                (
                    (x in inner_non_seqs_set)
                    or (x.owner in to_remove_set)
                    or isinstance(x, Constant)
                )
                for x in nd.inputs
            )
            # We can (supposedly) do this because the assumption is that a
            # `ViewOp` or `DeepCopyOp` will be just at the end of the
            # function and not somewhere in the middle
            and not isinstance(nd.op, aesara.compile.ViewOp)
            and not isinstance(nd.op, aesara.compile.DeepCopyOp)
        ):
            # We have a candidate node to remove from the inner-graph

            # Step 1. Reconstruct the node using the relevant outer-inputs.
            #
            # More specifically, the node's current inputs are either
            # a) inner-graph input place-holders for non-sequences,
            # b) the outputs of other nodes being pushed out of the inner-graph,
            # c) or constants.
            to_remove_set.add(nd)
            new_inputs = []
            for old_input in nd.inputs:
                if old_input in inner_non_seqs_set:
                    # This is case a), so we want to use the corresponding
                    # outer-graph input as the input to our new pushed-out node
                    _idx = inner_non_seqs_map[old_input]
                    new_input = outer_non_seqs[_idx]
                elif old_input in to_replace_set:
                    # This is case b), so we want to use the new pushed-out node
                    # as the input to this new pushed-out node
                    new_input = replace_with_out[to_replace_map[old_input]]
                else:
                    assert isinstance(old_input, Constant)
                    new_input = old_input

                new_input = old_input.type.filter_variable(new_input)
                new_inputs.append(new_input)

            pushed_out_node = nd.op.make_node(*new_inputs)

            if config.compute_test_value != "off":
                compute_test_value(pushed_out_node)

            # Step 2. Create variables to replace the old outputs of the node
            # that we're pushing out of the inner-graph
            for idx, y in enumerate(nd.outputs):
                y_place_holder = y.clone()
                # y_place_holder = safe_new(y, "_replace")
                add_to_replace(y)
                replace_with_in.append(y_place_holder)
                assert isinstance(y, type(pushed_out_node.outputs[idx]))
                replace_with_out.append(pushed_out_node.outputs[idx])

    # We need to check all candidate replacements and choose those that
    # make sense for us
    # Step 1. which elements of `to_replace` are used by remaining
    # components of the inner function
    clean_to_replace = []
    clean_replace_with_in = []
    clean_replace_with_out = []
    existent_nodes = [nd for nd in local_fgraph_topo if nd not in to_remove_set]
    existent_nodes_set = set(existent_nodes)

    to_keep_set = set()
    for nd in existent_nodes:
        to_keep_set.update(nd.inputs)

    for out, idx in to_replace_map.items():
        if (  # If types are different, conversion Op will be inserted,
            # and it may trigger an infinite loop.
            out.type.is_super(replace_with_in[idx].type)
            and out in to_keep_set
            and out.owner not in existent_nodes_set
        ):
            clean_to_replace.append(out)
            clean_replace_with_in.append(replace_with_in[idx])
            clean_replace_with_out.append(replace_with_out[idx])

    if len(clean_to_replace) > 0:
        # We can finally put an end to all this madness
        givens = {}
        nw_outer = []
        nw_inner = []
        for to_repl, repl_in, repl_out in zip(
            clean_to_replace, clean_replace_with_in, clean_replace_with_out
        ):
            if isinstance(repl_out, Constant):
                repl_in = repl_out
            else:
                nw_inner.append(repl_in)
                nw_outer.append(repl_out)
            givens[to_repl] = repl_in

        op_outs = clone_replace(node_outputs, replace=givens)
        op_ins = node_inputs + nw_inner

        new_info = dataclasses.replace(
            op.info, n_non_seqs=op.info.n_non_seqs + len(nw_outer)
        )

        # Reconstruct node
        nwScan = Scan(
            op_ins,
            op_outs,
            new_info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )

        # Do not call make_node for test_value
        nw_node = nwScan(*(node.inputs + nw_outer), return_list=True)[0].owner

        replacements = dict(zip(node.outputs, nw_node.outputs))
        replacements["remove"] = [node]
        return replacements
    elif not to_keep_set:
        # Nothing in the inner graph should be kept
        replace_with = {}
        for out, idx in to_replace_map.items():
            if out in local_fgraph_outs_set:
                x = node.outputs[local_fgraph_outs_map[out]]
                y = replace_with_out[idx]
                y_shape = [shp for shp in y.shape]
                replace_with[x] = at.alloc(y, node.inputs[0], *y_shape)

        # We need to add one extra dimension to the outputs
        # because the scan op expects for a tensor3, to which an
        # subtensor is applied that takes only the last element
        if replace_with:
            if len(node.outputs) == len(replace_with):
                # Every output of the node has a replacement, the Scan
                # node can be removed from the graph
                replace_with["remove"] = [node]
                return replace_with
            else:
                # The node has some outputs for which no replacement has
                # been established. This can occur for outputs that are
                # not produced by apply nodes (since the optimizations
                # only visits apply nodes) such as constants or inputs
                # passed directly as outputs. The replacements can be
                # performed but the Scan node can't be removed at this
                # point.
                return replace_with

    else:
        return False


@local_optimizer([Scan])
def push_out_seq_scan(fgraph, node):
    r"""Push out the variables inside the `Scan` that depend only on constants and sequences.

    This optimization resembles `push_out_non_seq_scan` but it tries to push--out of
    the inner function--the computation that only relies on sequence and
    non-sequence inputs. The idea behind this optimization is that, when it is
    possible to do so, it is generally more computationally efficient to perform
    a single operation on a large tensor rather then perform that same operation
    many times on many smaller tensors. In many cases, this optimization can
    increase memory usage but, in some specific cases, it can also decrease it.
    """
    if not isinstance(node.op, Scan):
        return False

    node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs

    local_fgraph_topo = io_toposort(node_inputs, node_outputs)
    local_fgraph_outs_set = set(node_outputs)
    local_fgraph_outs_map = {v: k for k, v in enumerate(node_outputs)}

    to_remove_set = set()
    to_replace_set = set()
    to_replace_map = {}

    def add_to_replace(y):
        to_replace_set.add(y)
        to_replace_map[y] = add_to_replace.n
        add_to_replace.n += 1

    add_to_replace.n = 0

    replace_with_in = []
    replace_with_out = []

    op = node.op
    # Construct the list of non_sequences to simplify a few things
    inner_non_seqs = op.inner_non_seqs(node_inputs)
    inner_non_seqs_set = set(inner_non_seqs)
    inner_non_seqs_map = {v: k for k, v in enumerate(inner_non_seqs)}

    outer_non_seqs = op.outer_non_seqs(node.inputs)
    inner_seqs = op.inner_seqs(node_inputs)
    inner_seqs_set = set(inner_seqs)
    inner_seqs_map = {v: k for k, v in enumerate(inner_seqs)}

    outer_seqs = op.outer_seqs(node.inputs)
    assert len(inner_non_seqs) == len(outer_non_seqs)
    assert len(inner_seqs) == len(outer_seqs)

    for nd in local_fgraph_topo:
        if (
            nd not in to_remove_set
            and all(
                (x in inner_non_seqs_set)
                or (x.owner in to_remove_set)
                or isinstance(x, Constant)
                or (x in inner_seqs_set)
                for x in nd.inputs
            )
            and isinstance(nd.op, Elemwise)
        ):

            outside_ins = []
            depends_on_seqs = False

            for x in nd.inputs:
                if x in inner_non_seqs_set:
                    _idx = inner_non_seqs_map[x]
                    new_input = outer_non_seqs[_idx]
                elif x in inner_seqs_set:
                    new_input = outer_seqs[inner_seqs_map[x]]
                    depends_on_seqs = True
                elif x in to_replace_set:
                    new_input = replace_with_out[to_replace_map[x]]
                    depends_on_seqs = True
                else:
                    assert isinstance(x, Constant)
                    new_input = x

                outside_ins.append(new_input)

            if not depends_on_seqs:
                # Removing this node from the inner graph of scan
                # should be handled by the PushOutNonSeqScan
                # optimization. The current optimization only tries
                # to pull sequence-dependant computation out of
                # scan.
                continue

            to_remove_set.add(nd)

            # Do not call make_node for test_value
            nw_outer_node = nd.op.make_node(*outside_ins)

            if config.compute_test_value != "off":
                compute_test_value(nw_outer_node)

            # Step 2. Create variables for replacements
            for idx, y in enumerate(nd.outputs):
                y_place_holder = safe_new(y, "_replace")
                add_to_replace(y)
                replace_with_in.append(y_place_holder)
                replace_with_out.append(nw_outer_node.outputs[idx])

        elif (
            nd not in to_remove_set
            and isinstance(nd.op, DimShuffle)
            and (nd.inputs[0] in inner_seqs_set or nd.inputs[0].owner in to_remove_set)
        ):

            to_remove_set.add(nd)
            x = nd.inputs[0]
            if x in inner_seqs_set:
                outside_ins = outer_seqs[inner_seqs_map[x]]
            elif x in to_replace_set:
                outside_ins = replace_with_out[to_replace_map[x]]
            new_ord = (0,)
            for old_ord in nd.op.new_order:
                if old_ord == "x":
                    new_ord += (old_ord,)
                else:
                    new_ord += (old_ord + 1,)
            new_outer = outside_ins.dimshuffle(new_ord)
            y = nd.outputs[0]
            y_place_holder = safe_new(y, "_replace")
            add_to_replace(y)
            replace_with_in.append(y_place_holder)
            replace_with_out.append(new_outer)

            if hasattr(new_outer.tag, "test_value"):
                new_sh = new_outer.tag.test_value.shape
                ref_sh = (outside_ins.tag.test_value.shape[0],)
                ref_sh += nd.outputs[0].tag.test_value.shape
                assert new_sh == ref_sh

    # We need to check all candidate replacements and choose those that
    # make sense for us
    # Step 1. which elements of `to_replace` are used by remaining
    # components of the inner function
    clean_to_replace = []
    clean_replace_with_in = []
    clean_replace_with_out = []

    existent_nodes = [nd for nd in local_fgraph_topo if nd not in to_remove_set]
    existent_nodes_set = set(existent_nodes)

    to_keep_set = set()
    for nd in existent_nodes:
        to_keep_set.update(nd.inputs)

    for out, idx in to_replace_map.items():
        if (
            out in to_keep_set
            and out.owner not in existent_nodes_set
            and
            # If types are different, conversion Op will be inserted,
            # and it may trigger an infinite loop.
            out.type.is_super(replace_with_in[idx].type)
        ):

            clean_to_replace.append(out)
            clean_replace_with_in.append(replace_with_in[idx])
            clean_replace_with_out.append(replace_with_out[idx])

    if len(clean_to_replace) > 0:
        # We can finally put an end to all this madness
        givens = {}
        nw_outer = []
        nw_inner = []
        for to_repl, repl_in, repl_out in zip(
            clean_to_replace, clean_replace_with_in, clean_replace_with_out
        ):
            if isinstance(repl_out, Constant):
                repl_in = repl_out
            else:
                nw_inner.append(repl_in)
                nw_outer.append(repl_out)

            givens[to_repl] = repl_in

        op_outs = clone_replace(node_outputs, replace=givens)
        op_ins = nw_inner + node_inputs

        # Reconstruct node
        nw_info = dataclasses.replace(op.info, n_seqs=op.info.n_seqs + len(nw_inner))
        nwScan = Scan(
            op_ins,
            op_outs,
            nw_info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )
        # Do not call make_node for test_value
        nw_node = nwScan(
            *(node.inputs[:1] + nw_outer + node.inputs[1:]),
            return_list=True,
        )[0].owner

        replacements = dict(zip(node.outputs, nw_node.outputs))
        replacements["remove"] = [node]
        return replacements

    elif not to_keep_set and not op.info.as_while and not op.outer_mitmot(node.inputs):
        # Nothing in the inner graph should be kept
        replace_with = {}
        for out, idx in to_replace_map.items():
            if out in local_fgraph_outs_set:
                x = node.outputs[local_fgraph_outs_map[out]]
                _y = replace_with_out[idx]
                ls = node_outputs
                if out in op.inner_mitsot_outs(ls):
                    odx = op.inner_mitsot_outs(ls).index(out)
                    inp = op.outer_mitsot(node.inputs)[odx]
                    st = abs(np.min(op.info.mit_sot_in_slices))
                    y = set_subtensor(inp[st:], _y)
                elif out in op.inner_sitsot_outs(ls):
                    odx = op.inner_sitsot_outs(ls).index(out)
                    inp = op.outer_sitsot(node.inputs)[odx]
                    y = set_subtensor(inp[1:], _y)
                elif out in op.inner_nitsot_outs(ls):
                    y = _y
                else:
                    y = _y[-1]
                replace_with[x] = y

        # We need to add one extra dimension to the outputs
        if replace_with and len(replace_with) == len(node.outputs):
            replacements = dict(replace_with.items())
            replacements["remove"] = [node]
            return replacements
    else:
        return False


def inner_sitsot_only_last_step_used(
    fgraph: FunctionGraph, var: Variable, scan_args: ScanArgs
) -> bool:
    """
    Given a inner nit-sot output of `Scan`, return ``True`` iff the outer
    nit-sot output has only one client and that client is a `Subtensor`
    instance that takes only the last step (last element along the first
    axis).
    """
    idx = scan_args.inner_out_sit_sot.index(var)
    outer_var = scan_args.outer_out_sit_sot[idx]

    if len(fgraph.clients[outer_var]) == 1:
        client = fgraph.clients[outer_var][0][0]
        if isinstance(client, Apply) and isinstance(client.op, Subtensor):
            lst = get_idx_list(client.inputs, client.op.idx_list)
            if len(lst) == 1 and at.extract_constant(lst[0]) == -1:
                return True

    return False


def get_outer_ndim(var: Variable, scan_args: ScanArgs) -> int:
    """Determine the number of dimension a variable would have if it was pushed out of a `Scan`."""
    assert isinstance(var.type, HasShape)

    if var in scan_args.inner_in_non_seqs or isinstance(var, Constant):
        outer_ndim = var.type.ndim
    else:
        outer_ndim = var.type.ndim + 1

    return outer_ndim


def push_out_inner_vars(
    fgraph: FunctionGraph,
    inner_vars: List[Variable],
    old_scan_node: Apply,
    old_scan_args: ScanArgs,
) -> Tuple[List[Variable], ScanArgs, Dict[Variable, Variable]]:

    tmp_outer_vars: List[Optional[Variable]] = []
    new_scan_node = old_scan_node
    new_scan_args = old_scan_args
    replacements: Dict[Variable, Variable] = {}

    # For the inner_vars that already exist in the outer graph,
    # simply obtain a reference to them
    for idx in range(len(inner_vars)):

        var = inner_vars[idx]

        new_outer_var: Optional[Variable] = None

        if var in old_scan_args.inner_in_seqs:
            idx_seq = old_scan_args.inner_in_seqs.index(var)
            new_outer_var = old_scan_args.outer_in_seqs[idx_seq]

        elif var in old_scan_args.inner_in_non_seqs:
            idx_non_seq = old_scan_args.inner_in_non_seqs.index(var)
            new_outer_var = old_scan_args.outer_in_non_seqs[idx_non_seq]

        elif isinstance(var, Constant):
            new_outer_var = var

        elif var in old_scan_args.inner_out_nit_sot:
            idx_nitsot = old_scan_args.inner_out_nit_sot.index(var)
            new_outer_var = old_scan_args.outer_out_nit_sot[idx_nitsot]

        tmp_outer_vars.append(new_outer_var)

    # For the inner_vars that don't already exist in the outer graph, add
    # them as new nitsot outputs to the scan node.
    idx_add_as_nitsots = [i for i, v in enumerate(tmp_outer_vars) if v is None]
    add_as_nitsots = [inner_vars[idx] for idx in idx_add_as_nitsots]

    new_outs: List[Variable] = []

    if len(add_as_nitsots) > 0:

        new_scan_node, replacements = add_nitsot_outputs(
            fgraph, old_scan_node, old_scan_args, add_as_nitsots
        )

        assert isinstance(new_scan_node.op, Scan)

        new_scan_args = ScanArgs(
            new_scan_node.inputs,
            new_scan_node.outputs,
            new_scan_node.op.inner_inputs,
            new_scan_node.op.inner_outputs,
            new_scan_node.op.info,
        )

        new_outs = new_scan_args.outer_out_nit_sot[-len(add_as_nitsots) :]

    outer_vars: List[Variable] = []

    for i, v in enumerate(tmp_outer_vars):
        if i in idx_add_as_nitsots:
            outer_vars.append(new_outs.pop(0))
        else:
            assert v is not None
            outer_vars.append(v)

    return outer_vars, new_scan_args, replacements


def add_nitsot_outputs(
    fgraph: FunctionGraph,
    old_scan_node: Apply,
    old_scan_args: ScanArgs,
    new_outputs_inner,
) -> Tuple[Apply, Dict[Variable, Variable]]:

    assert isinstance(old_scan_node.op, Scan)

    nb_new_outs = len(new_outputs_inner)

    # Create the initial values for the new nitsot outputs
    # (the initial value is the nb of steps to store. For a nistot,
    # it should be the number of steps performed by scan)
    new_nitsots_initial_value = [old_scan_node.inputs[0] for i in range(nb_new_outs)]

    # Create the `ScanArgs` corresponding to the new `Scan` `Op` to create
    new_scan_args = copy.copy(old_scan_args)
    new_scan_args.inner_out_nit_sot.extend(new_outputs_inner)
    new_scan_args.outer_in_nit_sot.extend(new_nitsots_initial_value)

    assert isinstance(old_scan_node.op, Scan)

    # Create the `Scan` `Op` from the `ScanArgs`
    new_scan_op = Scan(
        new_scan_args.inner_inputs,
        new_scan_args.inner_outputs,
        new_scan_args.info,
        mode=old_scan_node.op.mode,
        profile=old_scan_node.op.profile,
        truncate_gradient=old_scan_node.op.truncate_gradient,
        # TODO: This seems questionable
        name=old_scan_node.op.name,
        allow_gc=old_scan_node.op.allow_gc,
    )

    # Create the Apply node for the scan op
    new_scan_outs = new_scan_op(*new_scan_args.outer_inputs, return_list=True)
    assert isinstance(new_scan_outs, list)
    new_scan_node = new_scan_outs[0].owner
    assert new_scan_node is not None

    # Modify the outer graph to make sure the outputs of the new scan are
    # used instead of the outputs of the old scan
    new_node_new_outputs_idx = len(old_scan_args.outer_outputs) - len(
        old_scan_args.outer_out_shared
    )

    new_node_old_outputs = (
        new_scan_node.outputs[:new_node_new_outputs_idx]
        + new_scan_node.outputs[new_node_new_outputs_idx + nb_new_outs :]
    )

    # TODO FIXME:
    # replacements = dict(zip(old_scan_node.outputs, new_node_old_outputs))
    # replacements["remove"] = [old_scan_node]
    # return new_scan_node, replacements
    fgraph.replace_all_validate_remove(  # type: ignore
        list(zip(old_scan_node.outputs, new_node_old_outputs)),
        remove=[old_scan_node],
        reason="scan_pushout_add",
    )
    return new_scan_node, {}


@local_optimizer([Scan])
def push_out_add_scan(fgraph, node):
    r"""Push `Add` operations performed at the end of the inner graph to the outside.

    Like `push_out_seq_scan`, this optimization aims to replace many operations
    on small tensors by few operations on large tensors. It can also lead to
    increased memory usage.
    """
    # Don't perform the optimization on `as_while` `Scan`s. Because these
    # `Scan`s don't run for a predetermined number of steps, handling them is
    # more complicated and this optimization doesn't support it at the moment.
    if not (isinstance(node.op, Scan) and not node.op.info.as_while):
        return False

    op = node.op

    # Use `ScanArgs` to parse the inputs and outputs of scan for ease of
    # use
    args = ScanArgs(
        node.inputs, node.outputs, op.inner_inputs, op.inner_outputs, op.info
    )

    clients = {}
    local_fgraph_topo = io_toposort(
        args.inner_inputs, args.inner_outputs, clients=clients
    )

    for nd in local_fgraph_topo:
        if (
            isinstance(nd.op, Elemwise)
            and isinstance(nd.op.scalar_op, aes.Add)
            and nd.out in args.inner_out_sit_sot
            and inner_sitsot_only_last_step_used(fgraph, nd.out, args)
        ):

            # Ensure that one of the input to the add is the output of
            # the add from a previous iteration of the inner function
            sitsot_idx = args.inner_out_sit_sot.index(nd.out)
            if args.inner_in_sit_sot[sitsot_idx] in nd.inputs:

                # Ensure that the other input to the add is a dot product
                # between 2 matrices which will become a tensor3 and a
                # matrix if pushed outside of the scan. Also make sure
                # that the output of the Dot is ONLY used by the 'add'
                # otherwise doing a Dot in the outer graph will only
                # duplicate computation.

                sitsot_in_idx = nd.inputs.index(args.inner_in_sit_sot[sitsot_idx])

                # 0 if sitsot_in_idx==1, 1 if sitsot_in_idx==0
                dot_in_idx = 1 - sitsot_in_idx

                dot_input = nd.inputs[dot_in_idx]

                if (
                    dot_input.owner is not None
                    and isinstance(dot_input.owner.op, Dot)
                    and len(clients[dot_input]) == 1
                    and dot_input.owner.inputs[0].ndim == 2
                    and dot_input.owner.inputs[1].ndim == 2
                    and get_outer_ndim(dot_input.owner.inputs[0], args) == 3
                    and get_outer_ndim(dot_input.owner.inputs[1], args) == 3
                ):

                    # The optimization can be be applied in this case.

                    # Move out of scan the two inputs to the Dot and
                    # perform a dot outside of scan on these two inputs
                    inner_dot_inputs = nd.inputs[dot_in_idx].owner.inputs
                    (
                        outer_dot_inputs,
                        new_scan_args,
                        replacements,
                    ) = push_out_inner_vars(fgraph, inner_dot_inputs, node, args)

                    # Collapse some of the dimensions of the tensors
                    # so that they become matrices. This is because a
                    # dot is usually faster on two large matrices than
                    # a bunch of small ones
                    outer_dot_inputs[0] = at.flatten(
                        outer_dot_inputs[0].dimshuffle(1, 0, 2), ndim=2
                    )

                    shape_input1 = shape(outer_dot_inputs[1])
                    outer_dot_inputs[1] = outer_dot_inputs[1].reshape(
                        (shape_input1[0] * shape_input1[1], shape_input1[2])
                    )

                    # Perform the dot on the newly obtained matrices and
                    # add the initial value
                    outer_dot_output = dot(*outer_dot_inputs)
                    init_value = new_scan_args.outer_in_sit_sot[sitsot_idx][0]
                    replacement = outer_dot_output + init_value

                    # Alter the outer graph to use the output of the
                    # external Dot instead of the output of scan
                    # Modify the outer graph to add the outer Dot
                    outer_sitsot = new_scan_args.outer_out_sit_sot[sitsot_idx]
                    subtensor_node = fgraph.clients[outer_sitsot][0][0]
                    outer_sitsot_last_step = subtensor_node.outputs[0]

                    replacements[outer_sitsot_last_step] = replacement
                    return replacements

    return False


class ScanInplaceOptimizer(GlobalOptimizer):
    """Make `Scan`s perform in-place.

    This optimization attempts to make `Scan` compute its recurrent outputs inplace
    on the input tensors that contain their initial states. This optimization can
    improve runtime performance as well as reduce memory usage.

    """

    alloc_ops = (Alloc, AllocEmpty)
    """
    Classes that represent operation that allocate new memory and that the
    optimization should duplicate so it can operate inplace on them.
    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())
        fgraph.attach_feature(DestroyHandler())

    def attempt_scan_inplace(
        self, fgraph: FunctionGraph, node: Apply[Scan], output_indices: List[int]
    ) -> Optional[Apply]:
        """Attempt to replace a `Scan` node by one which computes the specified outputs inplace.

        Parameters
        ----------
        fgraph
            Function graph in which to attempt the replacement
        node
            Scan node to replace by an inplace version
        output_indices
            Indices of the outputs to attempt to compute inplace
        """

        op = node.op

        # inputs corresponding to sequences and n_steps
        ls_begin = node.inputs[: 1 + op.info.n_seqs]
        ls = op.outer_mitmot(node.inputs)
        ls += op.outer_mitsot(node.inputs)
        ls += op.outer_sitsot(node.inputs)
        ls_end = op.outer_shared(node.inputs)
        ls_end += op.outer_nitsot(node.inputs)
        ls_end += op.outer_non_seqs(node.inputs)

        # In `ls`, duplicate any input which has more than one client and is
        # the output of an eligible allocation op
        for i in range(len(ls)):
            inp = ls[i]
            if (
                len(fgraph.clients[inp]) > 1
                and inp.owner
                and isinstance(inp.owner.op, self.alloc_ops)
            ):
                new_lsi = inp.owner.op.make_node(*inp.owner.inputs)

                if config.compute_test_value != "off":
                    compute_test_value(new_lsi)

                new_lsi_out = new_lsi.outputs

                if len(new_lsi_out) == 1:
                    new_lsi_out = new_lsi_out[0]

                ls[i] = new_lsi_out

        n_outs = len(ls)
        for idx in range(n_outs):
            if ls[idx] in ls[:idx]:
                ls[idx] = deep_copy_op(ls[idx])

        inputs = ls_begin + ls + ls_end

        new_op = op.clone()

        destroy_map = op.destroy_map.copy()
        for out_idx in output_indices:
            destroy_map[out_idx] = [out_idx + 1 + op.info.n_seqs]

        new_op.destroy_map = destroy_map

        # Do not call make_node for test_value
        new_outs = new_op(*inputs, return_list=True)

        assert isinstance(new_outs, list)

        try:
            # TODO FIXME: We need to stop using this approach (i.e. attempt
            # in-place replacements and wait for downstream failures to revert
            # the changes).  It prevents us from making smart, clear
            # rewrites and it adds a lot of unnecessary overhead that
            # involves dealing with inconsistent graphs.
            # This whole rewrite should be a simple local rewrite, but, because
            # of this awful approach, it can't be.
            fgraph.replace_all_validate_remove(  # type: ignore
                list(zip(node.outputs, new_outs)),
                remove=[node],
                reason="scan_make_inplace",
            )
            return cast(Apply[Scan], new_outs[0].owner)
        except InconsistencyError:
            # Failed moving output to be computed inplace
            return None

    def apply(self, fgraph):

        for scan_idx, original_node in enumerate(reversed(fgraph.toposort())):

            if not isinstance(original_node.op, Scan):
                continue

            # First attempt to make the Scan compute inplace every recurrent
            # output that seems like it could be computed inplace. If that
            # fails, go through these outputs individually, trying each of
            # them.
            op = original_node.op
            n_outs = op.info.n_mit_mot + op.info.n_mit_sot + op.info.n_sit_sot

            # Generate a list of outputs on which the node could potentially
            # operate inplace.
            out_indices = []
            for out_idx in range(n_outs):
                inp_idx = 1 + op.info.n_seqs + out_idx
                inp = original_node.inputs[inp_idx]

                # If the input is from an eligible allocation node, attempt to
                # be inplace on it, even if other nodes are modifying it
                # inplace.
                if inp.owner and isinstance(inp.owner.op, self.alloc_ops):
                    out_indices.append(out_idx)
                    continue

                # If the input is not from an eligible allocation node, only
                # attempt to be inplace on it if nothing else is currently
                # inplace on it.
                input_used_inplace = False
                for c in fgraph.clients[original_node.inputs[inp_idx]]:
                    client = c[0]

                    # Get the indices of this client's inputs on which it
                    # operates inplace
                    if client.op.destroy_map:
                        # This flattens the content of destroy_map.values()
                        # which is a list of lists
                        inplace_inp_indices = sum(client.op.destroy_map.values(), [])

                        inplace_inps = [client.inputs[i] for i in inplace_inp_indices]
                        if original_node.inputs[inp_idx] in inplace_inps:
                            input_used_inplace = True
                            break

                if not input_used_inplace:
                    out_indices.append(out_idx)

            if len(out_indices) == 0:
                continue

            new_node = self.attempt_scan_inplace(fgraph, original_node, out_indices)

            if new_node is None:
                # Making the scan compute all plausible recurrent outputs
                # inplace has failed. Attempt all plausible recurrent outputs
                # individually.

                new_node = original_node
                for pos in out_indices:
                    new_node = (
                        self.attempt_scan_inplace(fgraph, new_node, [pos]) or new_node
                    )


def select_min(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return minimum(x, y)


def select_max(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return maximum(x, y)


def sanitize(x):
    if x is None:
        return None
    else:
        return at.as_tensor_variable(x)


@local_optimizer([Scan])
def save_mem_new_scan(fgraph, node):
    r"""Graph optimizer that reduces scan memory consumption.

    This optimizations attempts to determine if a `Scan` node, during its execution,
    for any of its outputs, can get away with allocating a memory buffer that is
    large enough to contain some of the computed timesteps of that output but not
    all of them.

    By default, during the execution of a `Scan` node, memory buffers will be
    allocated to store the values computed for every output at every iteration.
    However, in some cases, there are outputs for which there is only really a
    need to store the most recent ``N`` values, not all of them.

    For instance, if a `Scan` node has a SITSOT output (last computed value is
    fed back as an input at the next iteration) and only the last timestep of
    that output is ever used in the outer function, the `ScanSaveMem` optimization
    could determine that there is no need to store all computed timesteps for
    that SITSOT output. Only the most recently computed timestep ever needs to
    be kept in memory.

    """
    if not isinstance(node.op, Scan):
        return False

    if hasattr(fgraph, "shape_feature"):
        shape_of = fgraph.shape_feature.shape_of
    else:
        # Each access to shape_of is in a try..except block in order to
        # use a default version when the variable is not in the shape_of
        # dictionary.
        shape_of = {}
    # 1. Initialization of variables
    # Note 1) We do not actually care about outputs representing shared
    # variables (those have no intermediate values) so it is safer to
    # ignore them and not change them in any way. To simplify the
    # optimizations I construct the variable ``c_outs`` ( that counts
    # outputs up to those we care) and the list ``init_l`` which for any
    # output we care says the length of its initial state. Note that
    # defining ``init_l`` for mit_mot sequences is a bit trickier but
    # it is safe to set it to 0
    op = node.op
    op_info = op.info
    c_outs = (
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
    )

    init_l = [0 for x in range(op_info.n_mit_mot)]
    init_l += [
        abs(min(v)) for v in chain(op_info.mit_sot_in_slices, op_info.sit_sot_in_slices)
    ]
    init_l += [0 for x in range(op_info.n_nit_sot)]
    # 2. Check the clients of each output and see for how many steps
    # does scan need to run

    # This comparison checks if there is any uncounted output, which
    # can only be an output corresponding to a shared variable

    # 2.1 Initialize
    # global_nsteps is a dictionary having two fields ( 'real' deals
    # with int values, 'sym' with symbolic ones) or None
    # given that a scan op has k outputs o_1, .. o_k and each
    # output has n_j clients c_1^1, c_1^2, .. c_1^{n_1}, c_2^1, ..,
    # global_nsteps is None if any of the clients is different
    # from a subtensor or its real and sym field equal to
    # max(c_i_j.idx_list[0].stop), meaning store up to which maximal
    # index(step) for any output scan actually needs to compute
    # In other words n_steps should be equal to this maximal !
    # Note: if we have a shared variable that gets updated at every step
    # of the loop, reducing the number of steps will affect the the
    # value of the shared variable after the loop so we need not to
    # change the number of steps in that case. To do this we set
    # global_nsteps to None which is seen as a flag that nothing needs
    # to be done
    assert len(node.outputs) >= c_outs
    if len(node.outputs) == c_outs:
        global_nsteps = {"real": -1, "sym": []}
    else:
        global_nsteps = None

    # Keeps track of the original slices that each client represent
    slices = [None for o in node.outputs]

    # A list for each output indicating how many intermediate values
    # should be stored. If negative it means none of the intermediate
    # values (i.e. the output can be removed since it is not used
    # afterwards in the computations), if 0 it means that all
    # intermediate values are required, otherwise is up to that number
    # of intermediate values
    # Note that for mit_mot outputs and shared outputs we can not change
    # the number of intermediate steps stored without affecting the
    # result of the op
    store_steps = [0 for o in range(op_info.n_mit_mot)]
    store_steps += [-1 for o in node.outputs[op_info.n_mit_mot : c_outs]]
    # Flag that says if an input has changed and we need to do something
    # or not
    flag_store = False

    # 2.2 Loop over the clients
    for i, out in enumerate(node.outputs[:c_outs]):
        # look at all its clients
        slices[i] = []
        for cl, _ in fgraph.clients[out]:

            # 2.1 outputs of the function
            # => output needs all its intermediate values
            if isinstance(cl, str):
                # if the node is actually an output, then
                # we need to store the entire thing
                global_nsteps = None
                slices[i] = None
                break
            # 2.2 non-subtensor nodes
            # => output needs all its intermediate values
            elif not isinstance(cl.op, Subtensor):
                global_nsteps = None
                slices[i] = None
                break
            # 2.3 subtensor nodes
            # => output might need to store just a subset of its values
            else:
                # 2.3.1 extract idx list of subtensor
                this_slice = get_idx_list(cl.inputs, cl.op.idx_list)
                if this_slice is None:
                    # if unable to extract idx_list
                    # => outputs needs all its intermediate values
                    global_nsteps = None
                    slices[i] = None
                    break

                # 2.3.2 extract the begin/end of the first dimension
                if i >= op_info.n_mit_mot:
                    try:
                        length = shape_of[out][0]
                    except KeyError:
                        length = node.inputs[0] + init_l[i]
                else:
                    try:
                        length = shape_of[out][0]
                    except KeyError:
                        length = out.shape[0]
                cf_slice = get_canonical_form_slice(this_slice[0], length)
                slices[i] += [(cf_slice, this_slice)]

                if isinstance(this_slice[0], slice) and this_slice[0].stop is None:
                    global_nsteps = None
                if isinstance(cf_slice[0], slice):
                    stop = at.extract_constant(cf_slice[0].stop)
                else:
                    stop = at.extract_constant(cf_slice[0]) + 1
                if stop == maxsize or stop == length:
                    stop = None
                else:
                    # there is a **gotcha** here ! Namely, scan returns an
                    # array that contains the initial state of the output
                    # as well. Which means that if have a initial state of
                    # length 3, and you look for 5 steps you get an output
                    # y of length 8. If you only use y[:5], this does not
                    # mean that you only need to loop for 5 steps but
                    # actually only for 2 steps ( the first 3 are the
                    # initial state)
                    stop = stop - init_l[i]

                # 2.3.3 we might get away with less number of steps
                if stop is not None and global_nsteps is not None:
                    # yes if it is a tensor
                    if isinstance(stop, Variable):
                        global_nsteps["sym"] += [stop]
                    # not if it is maxsize
                    elif isinstance(stop, int) and stop == maxsize:
                        global_nsteps = None
                    # yes if it is a int k, 0 < k < maxsize
                    elif isinstance(stop, int) and global_nsteps["real"] < stop:
                        global_nsteps["real"] = stop
                    # yes if it is a int k, 0 < k < maxsize
                    elif isinstance(stop, int) and stop > 0:
                        pass
                    # not otherwise
                    else:
                        global_nsteps = None

    # 2.3. Analyze global_nsteps to figure out for how many steps scan
    # needs to iterate
    if global_nsteps is not None:
        nw_steps = node.inputs[0]

        # there are some symbolic tensors that limit the number of
        # steps
        if len(global_nsteps["sym"]) == 0:
            sym_steps = None
        else:
            sym_steps = global_nsteps["sym"][0]
            for c in global_nsteps["sym"][1:]:
                sym_steps = maximum(sym_steps, c)

        if global_nsteps["real"] >= 0:
            real_steps = global_nsteps["real"]
        else:
            real_steps = None
        nw_steps = select_min(select_max(sym_steps, real_steps), node.inputs[0])

        # Make sure the ScanSaveMem optimization never makes the new
        # number of steps to be 0 (this could happen, for instance, if
        # the optimization detects that the outputs of the Scan go through
        # subtensor nodes that end up taking no elements) because Scan with
        # 0 iterations are not supported. Make sure the new number of steps
        # is at least 1.
        nw_steps = select_max(nw_steps, 1)
    else:
        nw_steps = node.inputs[0]
        global_nsteps = None

    # 2.4 Loop over the clients again now looking just to see how many
    # intermediate steps to store
    for i, out in enumerate(node.outputs[:c_outs]):
        # look at all its clients
        for cl, _ in fgraph.clients[out]:
            if isinstance(cl, str):
                store_steps[i] = 0
                break
            elif not isinstance(cl.op, Subtensor):
                store_steps[i] = 0
                break
            else:
                this_slice = get_idx_list(cl.inputs, cl.op.idx_list)
                if this_slice is None:
                    store_steps[i] = 0
                    break

                if isinstance(this_slice[0], slice) and this_slice[0].start is None:
                    store_steps[i] = 0
                    break

                if i > op_info.n_mit_mot:
                    length = node.inputs[0] + init_l[i]
                else:
                    try:
                        length = shape_of[out][0]
                    except KeyError:
                        length = out.shape[0]
                cf_slice = get_canonical_form_slice(this_slice[0], length)

                if isinstance(cf_slice[0], slice):
                    start = at.extract_constant(cf_slice[0].start)
                else:
                    start = at.extract_constant(cf_slice[0])
                if start == 0 or store_steps[i] == 0:
                    store_steps[i] = 0
                else:
                    # The "+ 1" is because of the memory pre-allocation
                    # mechanism used to in the Scan op to reduce overhead.
                    # To prevent aliasing between the inputs and outputs
                    # of recurrent states, it requires that the buffer be
                    # large enough to that, the new state and the oldest
                    # tap needed don't occupy the sample place in the
                    # circular buffer. For now, this only needs to be done
                    # for mitsots and sitsots (because mitmots are not
                    # currently supported by the mechanism) and only if
                    # the pre-allocation mechanism is activated.
                    prealloc_outs = config.scan__allow_output_prealloc

                    first_mitsot_idx = op_info.n_mit_mot
                    last_sitsot_idx = (
                        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot - 1
                    )
                    preallocable_output = first_mitsot_idx <= i <= last_sitsot_idx

                    if prealloc_outs and preallocable_output:
                        pval = select_max(nw_steps - start + init_l[i], init_l[i] + 1)
                    else:
                        pval = select_max(nw_steps - start + init_l[i], init_l[i])

                    if store_steps[i] != -1:
                        pval = select_max(pval, store_steps[i])

                    store_steps[i] = pval
                    flag_store = True

    orphane_outs = [
        i for i, x in enumerate(store_steps) if isinstance(x, int) and (x < 0)
    ]
    flag_store = flag_store or (len(orphane_outs) > 0)
    # 3. is there anything to change ?
    if flag_store or global_nsteps is not None:
        # 3.1 initialize inputs for the new scan
        old_outputs = []
        nw_inputs = list(node.inputs)
        nw_inputs[0] = nw_steps

        # 3.2 check orphane outputs to see if we can eliminate any
        required, not_required = scan_can_remove_outs(node.op, orphane_outs)
        # 3.3. compose replace pairs for those nodes that need not
        # to store everything in memory ( or ar orphane and required
        # by the inner function .. )
        replaced_outs = []
        offset = 1 + op_info.n_seqs + op_info.n_mit_mot
        for idx, _val in enumerate(store_steps[op_info.n_mit_mot :]):
            i = idx + op_info.n_mit_mot
            if not (isinstance(_val, int) and _val <= 0 and i not in required):

                if idx + op_info.n_mit_mot in required:
                    val = 1
                else:
                    val = _val
                # If the memory for this output has been pre-allocated
                # before going into the scan op (by an alloc node)
                if idx < op_info.n_mit_sot + op_info.n_sit_sot:
                    # In case the input is still an alloc node, we
                    # actually have two options:
                    #   a) the input is a set_subtensor, in that case we
                    #      can replace the initial tensor by a slice,
                    #   b) it is not, and we simply take a slice of it.
                    # TODO: commit change below with Razvan
                    if (
                        nw_inputs[offset + idx].owner
                        and isinstance(nw_inputs[offset + idx].owner.op, IncSubtensor)
                        and isinstance(
                            nw_inputs[offset + idx].owner.op.idx_list[0], slice
                        )
                    ):

                        assert isinstance(
                            nw_inputs[offset + idx].owner.op, IncSubtensor
                        )
                        _nw_input = nw_inputs[offset + idx].owner.inputs[1]
                        cval = at.as_tensor_variable(val)
                        initl = at.as_tensor_variable(init_l[i])
                        tmp_idx = at.switch(cval < initl, cval + initl, cval - initl)
                        nw_input = expand_empty(_nw_input, tmp_idx)
                    else:
                        tmp = at.as_tensor_variable(val)
                        initl = at.as_tensor_variable(init_l[i])
                        tmp = maximum(tmp, initl)
                        nw_input = nw_inputs[offset + idx][:tmp]

                    nw_inputs[offset + idx] = nw_input
                    replaced_outs.append(op_info.n_mit_mot + idx)
                    odx = op_info.n_mit_mot + idx
                    old_outputs += [
                        (
                            odx,
                            [
                                x[0].outputs[0]
                                for x in fgraph.clients[node.outputs[odx]]
                            ],
                        )
                    ]
                # If there is no memory pre-allocated for this output
                elif idx < op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot:

                    pos = (
                        op_info.n_mit_mot
                        + idx
                        + op_info.n_seqs
                        + 1
                        + op_info.n_shared_outs
                    )
                    if nw_inputs[pos] == node.inputs[0]:
                        nw_inputs[pos] = val
                    odx = op_info.n_mit_mot + idx
                    replaced_outs.append(odx)
                    old_outputs += [
                        (
                            odx,
                            [
                                x[0].outputs[0]
                                for x in fgraph.clients[node.outputs[odx]]
                            ],
                        )
                    ]
        # 3.4. Recompute inputs for everything else based on the new
        # number of steps
        if global_nsteps is not None:
            for idx, val in enumerate(store_steps[op_info.n_mit_mot :]):
                if val == 0:
                    # val == 0 means that we want to keep all intermediate
                    # results for that state, including the initial values.
                    if idx < op_info.n_mit_sot + op_info.n_sit_sot:
                        in_idx = offset + idx
                        # Number of steps in the initial state
                        initl = init_l[op_info.n_mit_mot + idx]

                        # If the initial buffer has the form
                        # inc_subtensor(zeros(...)[...], _nw_input)
                        # we want to make the zeros tensor as small as
                        # possible (nw_steps + initl), and call
                        # inc_subtensor on that instead.
                        # Otherwise, simply take 0:(nw_steps+initl).
                        if (
                            nw_inputs[in_idx].owner
                            and isinstance(nw_inputs[in_idx].owner.op, IncSubtensor)
                            and isinstance(
                                nw_inputs[in_idx].owner.op.idx_list[0], slice
                            )
                        ):
                            _nw_input = nw_inputs[in_idx].owner.inputs[1]
                            nw_input = expand_empty(_nw_input, nw_steps)
                            nw_inputs[in_idx] = nw_input
                        else:
                            nw_input = nw_inputs[in_idx][: (initl + nw_steps)]

                    elif (
                        idx < op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
                    ):
                        in_idx = offset + idx + op_info.n_shared_outs
                        if nw_inputs[in_idx] == node.inputs[0]:
                            nw_inputs[in_idx] = nw_steps

        # 3.5 Remove unwanted orphane outputs
        (inps, outs, info, node_ins, compress_map) = compress_outs(
            op, not_required, nw_inputs
        )
        inv_compress_map = {}
        for k, v in compress_map.items():
            inv_compress_map[v] = k

        # 3.6 Compose the new scan
        # TODO: currently we don't support scan with 0 step. So
        # don't create one.
        if at.extract_constant(node_ins[0]) == 0:
            return False

        # Do not call make_node for test_value
        new_op = Scan(
            inps,
            outs,
            info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )
        new_outs = new_op(*node_ins, return_list=True)

        old_new = []
        # 3.7 Get replace pairs for those outputs that do not change
        # the number of intermediate steps stored
        for idx, sl in enumerate(slices):
            if global_nsteps and sl is not None and store_steps[idx] == 0:
                for hdx, cl in enumerate(fgraph.clients[node.outputs[idx]]):
                    cnf_slice, old_slices = sl[hdx]
                    # Sanitize the nw_slice by converting ints back into
                    # constants :) I only need to do this for the first
                    # slice since that is the only slice

                    if isinstance(cnf_slice[0], slice):
                        fslice = slice(
                            sanitize(cnf_slice[0].start),
                            sanitize(cnf_slice[0].stop),
                            sanitize(cnf_slice[0].step),
                        )
                    else:
                        fslice = sanitize(cnf_slice[0])

                    nw_slice = (fslice,) + tuple(old_slices[1:])
                    nw_pos = inv_compress_map[idx]

                    subtens = Subtensor(nw_slice)
                    # slice inputs
                    sl_ins = get_slice_elements(
                        nw_slice, lambda entry: isinstance(entry, Variable)
                    )
                    new_o = subtens(new_outs[nw_pos], *sl_ins)
                    if new_o.ndim > 0:
                        new_o = new_o[:: cnf_slice[1]]
                    replaced_outs.append(idx)
                    old_new += [(cl[0].outputs[0], new_o)]
        # 3.8. Get replace pairs for those outputs that change
        # the number of stored intermediate steps
        for pos, old_outs in old_outputs:
            if len(old_outs) > 0:
                nw_pos = compress_map[pos]
                for k, old in enumerate(old_outs):
                    # Get the correct slice
                    cnf_slice, old_slices = slices[pos][k]
                    if isinstance(cnf_slice[0], slice):
                        start = (
                            cnf_slice[0].start
                            - nw_steps
                            - init_l[pos]
                            + store_steps[pos]
                        )
                        if (
                            cnf_slice[0].stop is not None
                            and cnf_slice[0].stop != maxsize
                        ):
                            stop = (
                                cnf_slice[0].stop
                                - nw_steps
                                - init_l[pos]
                                + store_steps[pos]
                            )
                        else:
                            stop = None
                        nw_slice = (
                            slice(
                                sanitize(start),
                                sanitize(stop),
                                sanitize(cnf_slice[0].step),
                            ),
                        ) + tuple(old_slices[1:])

                    else:
                        position = (
                            cnf_slice[0] - nw_steps - init_l[pos] + store_steps[pos]
                        )

                        nw_slice = (sanitize(position),) + tuple(old_slices[1:])
                    subtens = Subtensor(nw_slice)
                    sl_ins = get_slice_elements(
                        nw_slice, lambda entry: isinstance(entry, Variable)
                    )
                    new_o = subtens(new_outs[nw_pos], *sl_ins)
                    if new_o.ndim > 0:
                        new_o = new_o[:: cnf_slice[1]]
                    old_new += [(old, new_o)]

        # 3.9. Get replace pairs for all other nodes
        if flag_store or global_nsteps is not None:
            for idx, o in enumerate(node.outputs):
                if not (idx in replaced_outs) and idx not in not_required:
                    nw_pos = compress_map[idx]
                    old_new += [(o, new_outs[nw_pos])]
            # Check if the new outputs depend on the old scan node
            old_scan_is_used = [
                is_in_ancestors(new.owner, node) for old, new in old_new
            ]
            if any(old_scan_is_used):
                return False

            replacements = dict(old_new)

            # remove = [old.owner for (old, new) in old_new]
            # As Fred suggested assert that also the old node is not in
            # the Graph as that will make things suboptimal
            # remove.append(node)
            replacements["remove"] = [node]

            return replacements

        return False


class ScanMerge(GlobalOptimizer):
    r"""Graph optimizer that merges different scan ops.

    This optimization attempts to fuse distinct `Scan` `Op`s into a single `Scan` `Op`
    that performs all the computation. The main advantage of merging `Scan` `Op`\s
    together comes from the possibility of both original `Op`\s having some
    computation in common. In such a setting, this computation ends up being done
    twice. The fused `Scan` `Op`, however, would only need to do it once and could
    therefore be more computationally efficient. Also, since every `Scan` node
    involves a certain overhead, at runtime, reducing the number of `Scan` nodes in
    the graph can improve performance.

    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def merge(self, nodes):

        if nodes[0].op.info.as_while:
            as_while = True
            condition = nodes[0].op.inner_outputs[-1]
        else:
            as_while = False

        # We keep the inner_ins and inner_outs of each original node separated.
        # To be able to recombine them in the right order after the clone,
        # we also need to split them by types (seq, mitmot, ...).
        # On the other hand, outer_ins, outer_outs and info are held together.
        inner_ins = [[] for nd in nodes]
        outer_ins = []
        inner_outs = [[] for nd in nodes]
        outer_outs = []

        def rename(ls, suffix):
            for k in ls:
                if k.name:
                    k.name += str(suffix)
            return ls

        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(rename(nd.op.inner_seqs(nd.op.inner_inputs), idx))
            outer_ins += rename(nd.op.outer_seqs(nd.inputs), idx)

        mit_mot_out_slices = ()

        mit_mot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(rename(nd.op.inner_mitmot(nd.op.inner_inputs), idx))
            inner_outs[idx].append(nd.op.inner_mitmot_outs(nd.op.inner_outputs))
            mit_mot_in_slices += nd.op.info.mit_mot_in_slices
            mit_mot_out_slices += nd.op.info.mit_mot_out_slices[: nd.op.info.n_mit_mot]
            outer_ins += rename(nd.op.outer_mitmot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitmot_outs(nd.outputs)

        mit_sot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(rename(nd.op.inner_mitsot(nd.op.inner_inputs), idx))
            inner_outs[idx].append(nd.op.inner_mitsot_outs(nd.op.inner_outputs))
            mit_sot_in_slices += nd.op.info.mit_sot_in_slices
            outer_ins += rename(nd.op.outer_mitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitsot_outs(nd.outputs)

        sit_sot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(rename(nd.op.inner_sitsot(nd.op.inner_inputs), idx))
            sit_sot_in_slices += tuple((-1,) for x in range(nd.op.info.n_sit_sot))
            inner_outs[idx].append(nd.op.inner_sitsot_outs(nd.op.inner_outputs))
            outer_ins += rename(nd.op.outer_sitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_sitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            inner_ins[idx].append(rename(nd.op.inner_shared(nd.op.inner_inputs), idx))
            outer_ins += rename(nd.op.outer_shared(nd.inputs), idx)

        for idx, nd in enumerate(nodes):
            # NitSot
            inner_outs[idx].append(nd.op.inner_nitsot_outs(nd.op.inner_outputs))
            outer_ins += rename(nd.op.outer_nitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_nitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            outer_outs += nd.op.outer_shared_outs(nd.outputs)
            inner_outs[idx].append(nd.op.inner_shared_outs(nd.op.inner_outputs))

        n_non_seqs = 0
        for idx, nd in enumerate(nodes):
            # Non Seqs
            node_inner_non_seqs = nd.op.inner_non_seqs(nd.op.inner_inputs)
            n_non_seqs += len(node_inner_non_seqs)
            inner_ins[idx].append(rename(node_inner_non_seqs, idx))
            outer_ins += rename(nd.op.outer_non_seqs(nd.inputs), idx)

        # Add back the number of steps
        outer_ins = [nodes[0].inputs[0]] + outer_ins

        if as_while:
            # add the condition, which was the one of nodes[0]
            inner_outs[0].append([condition])

        # Clone the inner graph of each node independently
        for idx, nd in enumerate(nodes):
            # concatenate all inner_ins and inner_outs of nd
            flat_inner_ins = sum(inner_ins[idx], [])
            flat_inner_outs = sum(inner_outs[idx], [])
            # clone
            flat_inner_ins, flat_inner_outs = reconstruct_graph(
                flat_inner_ins, flat_inner_outs
            )
            # split the new inner variables again in seq, mitmot, etc.
            new_inner_ins = []
            count = 0
            for nl in inner_ins[idx]:
                seq_len = len(nl)
                new_inner_ins.append(flat_inner_ins[count : (count + seq_len)])
                count += seq_len

            new_inner_outs = []
            count = 0
            for nl in inner_outs[idx]:
                seq_len = len(nl)
                new_inner_outs.append(flat_inner_outs[count : (count + seq_len)])
                count += seq_len

            inner_ins[idx] = new_inner_ins
            inner_outs[idx] = new_inner_outs

        # Flatten inner_ins and inner_outs so that all seqs are first,
        # then mitmot, etc.
        new_inner_ins = []
        new_inner_outs = []
        nb_ins_groups = len(inner_ins[0])
        nb_outs_groups = len(inner_outs[0])
        for idx, nd in enumerate(nodes):
            # All inner_ins should have the same length
            assert len(inner_ins[idx]) == nb_ins_groups

            # All inner_outs should have the same length, except if as_while,
            # in which case the first one should have one more element
            if as_while and idx > 0:
                assert len(inner_outs[idx]) == nb_outs_groups - 1
            else:
                assert len(inner_outs[idx]) == nb_outs_groups

        for gr_idx in range(nb_ins_groups):
            for idx, nd in enumerate(nodes):
                new_inner_ins += inner_ins[idx][gr_idx]

        for gr_idx in range(nb_outs_groups):
            for idx, nd in enumerate(nodes):
                if as_while and idx > 0 and gr_idx == (nb_outs_groups - 1):
                    # There is no condition on that node, skip it
                    pass
                else:
                    new_inner_outs += inner_outs[idx][gr_idx]

        info = ScanInfo(
            n_seqs=sum(nd.op.info.n_seqs for nd in nodes),
            mit_mot_in_slices=mit_mot_in_slices,
            mit_mot_out_slices=mit_mot_out_slices,
            mit_sot_in_slices=mit_sot_in_slices,
            sit_sot_in_slices=sit_sot_in_slices,
            n_nit_sot=sum(nd.op.info.n_nit_sot for nd in nodes),
            n_shared_outs=sum(nd.op.info.n_shared_outs for nd in nodes),
            n_non_seqs=n_non_seqs,
            as_while=as_while,
        )

        old_op = nodes[0].op
        new_op = Scan(
            new_inner_ins,
            new_inner_outs,
            info,
            mode=old_op.mode,
            profile=old_op.profile,
            truncate_gradient=old_op.truncate_gradient,
            allow_gc=old_op.allow_gc,
            name="&".join([nd.op.name for nd in nodes]),
        )
        new_outs = new_op(*outer_ins)

        if not isinstance(new_outs, (list, tuple)):
            new_outs = [new_outs]

        return list(zip(outer_outs, new_outs))

    def belongs_to_set(self, node, set_nodes):
        """
        This function checks if node `node` belongs to `set_nodes`, in the
        sense that it can be merged together with every other node in
        `set_nodes`. In order for two nodes to be mergeable, they have to go
        over the same number of steps, have the same condition (if any),
        have the same value for truncate_gradient, and have the same mode.
        Questionable, we should also consider profile ?

        """
        rep = set_nodes[0]
        if (
            rep.op.info.as_while != node.op.info.as_while
            or node.op.truncate_gradient != rep.op.truncate_gradient
            or node.op.mode != rep.op.mode
        ):
            return False

        nsteps = node.inputs[0]
        try:
            nsteps = int(get_scalar_constant_value(nsteps))
        except NotScalarConstantError:
            pass

        rep_nsteps = rep.inputs[0]
        try:
            rep_nsteps = int(get_scalar_constant_value(rep_nsteps))
        except NotScalarConstantError:
            pass

        if nsteps != rep_nsteps:
            return False

        # Check to see if it is an input of a different node
        for nd in set_nodes:
            if is_in_ancestors(node, nd) or is_in_ancestors(nd, node):
                return False

        if not node.op.info.as_while:
            return True
        cond = node.op.inner_outputs[-1]
        rep_cond = rep.op.inner_outputs[-1]
        return equal_computations(
            [cond], [rep_cond], node.op.inner_inputs, rep.op.inner_inputs
        )

    def apply(self, fgraph):
        # Collect all scan nodes ordered according to toposort
        scan_nodes = [nd for nd in fgraph.toposort() if isinstance(nd.op, Scan)]

        # All sets of possibly mergeable nodes
        all_sets = []

        for nd in scan_nodes:
            belongs_to_set_idx = -1
            for pos, subset in enumerate(all_sets):
                if self.belongs_to_set(nd, subset):
                    belongs_to_set_idx = pos
                    # It is possible that nd belongs to more than one subset.
                    # For instance, if we have 3 Scan nodes X, Y and Z, if Z
                    # depends on the output of X, then X and Z are incompatible
                    # and would create different subsets, but Y could be
                    # compatible with both X and Z. We choose the first one.
                    break

            if belongs_to_set_idx == -1:
                all_sets.append([nd])
            else:
                all_sets[belongs_to_set_idx].append(nd)

        for subset in all_sets:
            if len(subset) > 1:
                proposal = self.merge(subset)
                fgraph.replace_all_validate_remove(
                    proposal, remove=subset, reason="scan_merge"
                )


def has_duplicates(l):
    """
    Returns true if l has any duplicates (according to __eq__).

    """
    return len(set(l)) < len(l)


def make_equiv(lo, li):
    """
    Builds a dictionary of equivalences between inner inputs based on
    the equivalence of their corresponding outer inputs.

    """
    seeno = {}
    left = []
    right = []
    for o, i in zip(lo, li):
        if o in seeno:
            left += [i]
            right += [o]
        else:
            seeno[o] = i
    return left, right


@local_optimizer([Scan])
def scan_merge_inouts(fgraph, node):
    """
    This optimization attempts to merge a `Scan` `Op`'s identical outer inputs as well
    as merge its identical outer outputs (outputs that perform the same
    computation on the same inputs). This can reduce the amount of computation as
    well as result in a simpler graph for both the inner function and the outer
    function.
    """
    if not isinstance(node.op, Scan):
        return False

    # Do a first pass to merge identical external inputs.
    # Equivalent inputs will be stored in inp_equiv, then a new
    # scan node created without duplicates.
    a = ScanArgs(
        node.inputs,
        node.outputs,
        node.op.inner_inputs,
        node.op.inner_outputs,
        node.op.info,
    )

    inp_equiv = {}

    if has_duplicates(a.outer_in_seqs):
        new_outer_seqs = []
        new_inner_seqs = []
        for out_seq, in_seq in zip(a.outer_in_seqs, a.inner_in_seqs):
            if out_seq in new_outer_seqs:
                i = new_outer_seqs.index(out_seq)
                inp_equiv[in_seq] = new_inner_seqs[i]
            else:
                new_outer_seqs.append(out_seq)
                new_inner_seqs.append(in_seq)
        a.outer_in_seqs = new_outer_seqs
        a.inner_in_seqs = new_inner_seqs

    if has_duplicates(a.outer_in_non_seqs):
        new_outer_nseqs = []
        new_inner_nseqs = []
        for out_nseq, in_nseq in zip(a.outer_in_non_seqs, a.inner_in_non_seqs):
            if out_nseq in new_outer_nseqs:
                i = new_outer_nseqs.index(out_nseq)
                inp_equiv[in_nseq] = new_inner_nseqs[i]
            else:
                new_outer_nseqs.append(out_nseq)
                new_inner_nseqs.append(in_nseq)
        a.outer_in_non_seqs = new_outer_nseqs
        a.inner_in_non_seqs = new_inner_nseqs

    if len(inp_equiv) > 0:
        # do the replacement now. The rest will be left to ScanSaveMem
        inner_inputs = a.inner_inputs
        outer_inputs = a.outer_inputs
        info = a.info
        a_inner_outs = a.inner_outputs
        inner_outputs = clone_replace(a_inner_outs, replace=inp_equiv)

        new_op = Scan(
            inner_inputs,
            inner_outputs,
            info,
            mode=node.op.mode,
            profile=node.op.profile,
            truncate_gradient=node.op.truncate_gradient,
            # TODO: This seems questionable
            name=node.op.name,
            allow_gc=node.op.allow_gc,
        )
        outputs = new_op(*outer_inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        na = ScanArgs(
            outer_inputs,
            outputs,
            new_op.inner_inputs,
            new_op.inner_outputs,
            new_op.info,
        )
        remove = [node]
    else:
        na = a
        remove = []

    # Now that the identical external inputs have been merged, we do a new
    # loop in order to merge external outputs that compute the same things
    # from the same inputs.
    left = []
    right = []

    if has_duplicates(na.outer_in_shared):
        _left, _right = make_equiv(na.outer_in_shared, na.inner_in_shared)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_sit_sot):
        _left, _right = make_equiv(na.outer_in_sit_sot, na.inner_in_sit_sot)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_mit_mot):
        seen = {}
        for omm, imm, _sl in zip(
            na.outer_in_mit_mot, na.inner_in_mit_mot, na.mit_mot_in_slices
        ):
            sl = tuple(_sl)
            if (omm, sl) in seen:
                simm = seen[(omm, sl)]
                left += imm
                right += simm
            else:
                seen[(omm, sl)] = imm

    if has_duplicates(na.outer_in_mit_sot):
        seen = {}
        for oms, ims, _sl in zip(
            na.outer_in_mit_sot, na.inner_in_mit_sot, na.mit_sot_in_slices
        ):
            sl = tuple(_sl)
            if (oms, sl) in seen:
                sims = seen[(oms, sl)]
                left += ims
                right += sims
            else:
                seen[(oms, sl)] = ims

    def map_out(outer_i, inner_o, outer_o, seen):
        # Return the outer input corresponding to an
        # (outer input, inner output) pair. If we see that pair for the first
        # time, return the provided outer output. If an equivalent pair had
        # already been seen, return that one instead.
        # Note that we need to check that the outer input match as well,
        # because they could have different sizes, and the corresponding
        # outer outputs cannot be merged in that case.
        for s_outer_i, s_inner_o, s_outer_o in seen:
            if (
                equal_computations([inner_o], [s_inner_o], left, right)
                and outer_i == s_outer_i
            ):
                return s_outer_o
        seen.append((outer_i, inner_o, outer_o))
        return outer_o

    seen = []

    assert len(na.outer_in_nit_sot) == len(na.inner_out_nit_sot)
    assert len(na.inner_out_nit_sot) == len(na.outer_out_nit_sot)
    na.outer_out_nit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_nit_sot, na.inner_out_nit_sot, na.outer_out_nit_sot
        )
    ]

    seen = []
    assert len(na.outer_in_sit_sot) == len(na.inner_out_sit_sot)
    assert len(na.inner_out_sit_sot) == len(na.outer_out_sit_sot)
    na.outer_out_sit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_sit_sot, na.inner_out_sit_sot, na.outer_out_sit_sot
        )
    ]

    seen = []
    assert len(na.outer_in_mit_sot) == len(na.inner_out_mit_sot)
    assert len(na.inner_out_mit_sot) == len(na.outer_out_mit_sot)
    na.outer_out_mit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_mit_sot, na.inner_out_mit_sot, na.outer_out_mit_sot
        )
    ]

    seen = []
    new_outer_out_mit_mot = []
    assert len(na.outer_in_mit_mot) == len(na.inner_out_mit_mot)
    assert len(na.inner_out_mit_mot) == len(na.outer_out_mit_mot)
    assert len(na.outer_out_mit_mot) == len(na.mit_mot_out_slices)
    for outer_imm, inner_omm, outer_omm, osl in zip(
        na.outer_in_mit_mot,
        na.inner_out_mit_mot,
        na.outer_out_mit_mot,
        na.mit_mot_out_slices,
    ):
        for s_outer_imm, s_inner_omm, s_outer_omm, sosl in seen:
            if (
                osl == sosl
                and equal_computations(inner_omm, s_inner_omm, left, right)
                and outer_imm == s_outer_imm
            ):

                new_outer_out_mit_mot.append(s_outer_omm)
                break
        else:
            seen.append((outer_imm, inner_omm, outer_omm, osl))
            new_outer_out_mit_mot.append(outer_omm)
    na.outer_out_mit_mot = new_outer_out_mit_mot
    if remove:
        return dict([("remove", remove)] + list(zip(node.outputs, na.outer_outputs)))
    return na.outer_outputs


@local_optimizer([Scan])
def push_out_dot1_scan(fgraph, node):
    r"""
    This is another optimization that attempts to detect certain patterns of
    computation in a `Scan` `Op`'s inner function and move this computation to the
    outer graph.
    """
    if not isinstance(node.op, Scan):
        return False

    # Replace pattern of the form
    # x[t] = x[t-1] + dot(seq[t], value)
    # with Sequence.reshape((-1, seq.shape[2])) \dot Value
    # When seq[t] is a vector/matrix  and `value` is a matrix
    # Note that this works when only you need X[-1] in the end
    # and assumes dimshuffle are applied to vectors before calling dot
    op = node.op
    sitsot_ins = op.inner_sitsot(op.inner_inputs)
    sitsot_outs = op.inner_sitsot_outs(op.inner_outputs)
    outer_sitsot = op.outer_sitsot_outs(node.outputs)
    seqs = op.inner_seqs(op.inner_inputs)
    for inp, out, outer_out in zip(sitsot_ins, sitsot_outs, outer_sitsot):

        if (
            out.owner
            and isinstance(out.owner.op, Elemwise)
            and isinstance(out.owner.op.scalar_op, aes.Add)
            and inp in out.owner.inputs
            and len(fgraph.clients[outer_out]) == 1
            and not isinstance(fgraph.clients[outer_out][0][0], str)
            and isinstance(fgraph.clients[outer_out][0][0].op, Subtensor)
            and fgraph.clients[outer_out][0][0].op.idx_list == (-1,)
        ):

            x = out.owner.inputs[0]
            if x == inp:
                x = out.owner.inputs[1]
            # We need to check if x is the result of an outer product
            if (
                x.owner
                and isinstance(x.owner.op, Dot)
                and x.owner.inputs[0].ndim == 2
                and x.owner.inputs[1].ndim == 2
            ):

                # We need to check if any of the inputs are a sequence
                inp1 = x.owner.inputs[0]
                inp2 = x.owner.inputs[1]

                if inp1 in seqs or inp2 in seqs:
                    new_scan_out = inp1

                    if inp1 in seqs:
                        new_scan_out = inp2
                    idx = sitsot_outs.index(out)
                    # We've found our pattern and need to construct a new
                    # scan node to replace this one. For this we need to
                    # replace the sit_sot output with a nit_sot output

                    # First let us split all arguments according to their
                    # corresponding categories

                    inner_seqs = op.inner_seqs(op.inner_inputs)
                    outer_seqs = op.outer_seqs(node.inputs)
                    inner_mitmot = op.inner_mitmot(op.inner_inputs)
                    outer_mitmot = op.outer_mitmot(node.inputs)
                    inner_mitmot_outs = op.inner_mitmot_outs(op.inner_outputs)
                    inner_mitsot = op.inner_mitsot(op.inner_inputs)
                    outer_mitsot = op.outer_mitsot(node.inputs)
                    inner_mitsot_outs = op.inner_mitsot_outs(op.inner_outputs)
                    inner_sitsot = op.inner_sitsot(op.inner_inputs)
                    outer_sitsot = op.outer_sitsot(node.inputs)
                    inner_sitsot_outs = op.inner_sitsot_outs(op.inner_outputs)
                    outer_nitsot = op.outer_nitsot(node.inputs)
                    inner_nitsot_outs = op.inner_nitsot_outs(op.inner_outputs)
                    inner_shared = op.inner_shared(op.inner_inputs)
                    outer_shared = op.outer_shared(node.inputs)
                    inner_shared_outs = op.inner_shared_outs(op.inner_outputs)
                    inner_non_seqs = op.inner_non_seqs(op.inner_inputs)
                    outer_non_seqs = op.outer_non_seqs(node.inputs)

                    new_info = dataclasses.replace(
                        op.info,
                        sit_sot_in_slices=op.info.sit_sot_in_slices[:idx]
                        + op.info.sit_sot_in_slices[idx + 1 :],
                        n_nit_sot=op.info.n_nit_sot + 1,
                    )
                    inner_sitsot = inner_sitsot[:idx] + inner_sitsot[idx + 1 :]
                    outer_sitsot = outer_sitsot[:idx] + outer_sitsot[idx + 1 :]
                    inner_sitsot_outs = (
                        inner_sitsot_outs[:idx] + inner_sitsot_outs[idx + 1 :]
                    )
                    # add n_steps as the length
                    inner_nitsot_outs.append(new_scan_out)

                    _new_inner_inps = (
                        inner_seqs
                        + inner_mitmot
                        + inner_mitsot
                        + inner_sitsot
                        + inner_shared
                        + inner_non_seqs
                    )
                    _new_inner_outs = (
                        inner_mitmot_outs
                        + inner_mitsot_outs
                        + inner_sitsot_outs
                        + inner_nitsot_outs
                        + inner_shared_outs
                    )
                    new_inner_inps, new_inner_outs = reconstruct_graph(
                        _new_inner_inps, _new_inner_outs
                    )
                    new_op = Scan(
                        new_inner_inps,
                        new_inner_outs,
                        new_info,
                        mode=op.mode,
                        profile=op.profile,
                        truncate_gradient=op.truncate_gradient,
                        # TODO: This seems questionable
                        name=op.name,
                        allow_gc=op.allow_gc,
                    )
                    _scan_inputs = (
                        [node.inputs[0]]
                        + outer_seqs
                        + outer_mitmot
                        + outer_mitsot
                        + outer_sitsot
                        + outer_shared
                        + outer_nitsot
                        + [node.inputs[0]]
                        + outer_non_seqs
                    )

                    new_outs = new_op(*_scan_inputs)
                    if not isinstance(new_outs, (list, tuple)):
                        new_outs = [new_outs]

                    # We need now to pair correctly the new outputs
                    # with the old ones

                    outer_nitsot_outs = new_op.outer_nitsot_outs(new_outs)

                    _val = outer_nitsot_outs[-1]
                    outer_nitsot_outs = outer_nitsot_outs[:-1]
                    if inp1 in seqs:
                        _out_seq = op.outer_seqs(node.inputs)[seqs.index(inp1)]
                        # We need to clip the seq to the number of steps
                        _out_seq = _out_seq[: node.inputs[0]]
                        sh0 = _out_seq.shape[0]
                        sh1 = _out_seq.shape[1]
                        sh2 = _out_seq.shape[2]
                        out_seq = _out_seq.dimshuffle(1, 0, 2)
                        out_seq = out_seq.reshape((sh1, sh0 * sh2))
                        sh0 = _val.shape[0]
                        sh1 = _val.shape[1]
                        sh2 = _val.shape[2]

                        val = _val.reshape((sh0 * sh1, sh2))
                        new_out = dot(out_seq, val)
                    else:
                        _out_seq = op.outer_seqs(node.inputs)[seqs.index(inp2)]
                        out_seq = _out_seq.reshape(
                            (
                                _out_seq.shape[0] * _out_seq.shape[1],
                                _out_seq.shape[2],
                            )
                        )

                        val = _val.dimshuffle(1, 0, 2).reshape(
                            (_val.shape[1], _val.shape[0] * _val.shape[2])
                        )
                        new_out = dot(val, out_seq)

                    pos = node.outputs.index(outer_out)
                    old_new = list(zip(node.outputs[:pos], new_outs[:pos]))
                    old = fgraph.clients[node.outputs[pos]][0][0].outputs[0]
                    old_new.append((old, new_out))
                    old_new += list(zip(node.outputs[pos + 1 :], new_outs[pos:]))
                    replacements = dict(old_new)
                    replacements["remove"] = [node]
                    return replacements

    return False


# I've added an equilibrium because later scan optimization in the sequence
# can make it such that earlier optimizations should apply. However, in
# general I do not expect the sequence to run more then once
scan_eqopt1 = EquilibriumDB()
scan_seqopt1 = SequenceDB()
scan_eqopt2 = EquilibriumDB()

# scan_eqopt1 before ShapeOpt at 0.1
# This is needed to don't have ShapeFeature trac old Scan that we
# don't want to reintroduce.
optdb.register("scan_eqopt1", scan_eqopt1, "fast_run", "scan", position=0.05)
# We run before blas opt at 1.7 and specialize 2.0
# but after stabilize at 1.5. Should we put it before stabilize?
optdb.register("scan_eqopt2", scan_eqopt2, "fast_run", "scan", position=1.6)
# ScanSaveMem should execute only once per node.
optdb.register(
    "scan_save_mem",
    in2out(save_mem_new_scan, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=1.61,
)
optdb.register(
    "scan_make_inplace",
    ScanInplaceOptimizer(),
    "fast_run",
    "inplace",
    "scan",
    position=75,
)

scan_eqopt1.register("all_pushout_opt", scan_seqopt1, "fast_run", "scan", position=1)


scan_seqopt1.register(
    "scan_remove_constants_and_unused_inputs0",
    in2out(remove_constants_and_unused_inputs_scan, ignore_newtrees=True),
    "remove_constants_and_unused_inputs_scan",
    "fast_run",
    "scan",
    position=1,
)


scan_seqopt1.register(
    "scan_pushout_nonseqs_ops",
    in2out(push_out_non_seq_scan, ignore_newtrees=True),
    "fast_run",
    "scan",
    "scan_pushout",
    position=2,
)


scan_seqopt1.register(
    "scan_pushout_seqs_ops",
    in2out(push_out_seq_scan, ignore_newtrees=True),
    "fast_run",
    "scan",
    "scan_pushout",
    position=3,
)


scan_seqopt1.register(
    "scan_pushout_dot1",
    in2out(push_out_dot1_scan, ignore_newtrees=True),
    "fast_run",
    "more_mem",
    "scan",
    "scan_pushout",
    position=4,
)


scan_seqopt1.register(
    "scan_pushout_add",
    # TODO: Perhaps this should be an `EquilibriumOptimizer`?
    in2out(push_out_add_scan, ignore_newtrees=False),
    "fast_run",
    "more_mem",
    "scan",
    "scan_pushout",
    position=5,
)


scan_eqopt2.register(
    "constant_folding_for_scan2",
    in2out(basic_opt.constant_folding, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=1,
)


scan_eqopt2.register(
    "scan_remove_constants_and_unused_inputs1",
    in2out(remove_constants_and_unused_inputs_scan, ignore_newtrees=True),
    "remove_constants_and_unused_inputs_scan",
    "fast_run",
    "scan",
    position=2,
)


# after const merge but before stabilize so that we can have identity
# for equivalent nodes but we still have the chance to hoist stuff out
# of the scan later.
scan_eqopt2.register("scan_merge", ScanMerge(), "fast_run", "scan", position=4)

# After Merge optimization
scan_eqopt2.register(
    "scan_remove_constants_and_unused_inputs2",
    in2out(remove_constants_and_unused_inputs_scan, ignore_newtrees=True),
    "remove_constants_and_unused_inputs_scan",
    "fast_run",
    "scan",
    position=5,
)

scan_eqopt2.register(
    "scan_merge_inouts",
    in2out(scan_merge_inouts, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=6,
)

# After everything else
scan_eqopt2.register(
    "scan_remove_constants_and_unused_inputs3",
    in2out(remove_constants_and_unused_inputs_scan, ignore_newtrees=True),
    "remove_constants_and_unused_inputs_scan",
    "fast_run",
    "scan",
    position=8,
)

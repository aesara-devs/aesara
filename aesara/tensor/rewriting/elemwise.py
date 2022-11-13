import sys
from collections import defaultdict, deque
from functools import lru_cache
from typing import Any, Dict, Generator, List, Tuple
from typing import cast as typing_cast
from warnings import warn

import aesara
import aesara.scalar.basic as aes
from aesara import compile
from aesara.configdefaults import config
from aesara.graph import FunctionGraph
from aesara.graph.basic import (
    Apply,
    Constant,
    Variable,
    ancestors,
    clone_replace,
    io_toposort,
)
from aesara.graph.features import ReplaceValidate
from aesara.graph.rewriting.basic import GraphRewriter, copy_stack_trace, node_rewriter
from aesara.graph.rewriting.db import SequenceDB
from aesara.graph.utils import InconsistencyError
from aesara.tensor.basic import MakeVector, alloc, cast, get_scalar_constant_value
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.rewriting.basic import register_canonicalize, register_specialize
from aesara.tensor.shape import shape_padleft
from aesara.tensor.var import TensorConstant


class InplaceElemwiseOptimizer(GraphRewriter):
    r"""
    This is parameterized so that it works for `Elemwise` `Op`\s.
    """

    def __init__(self, OP):
        self.op = OP

    def add_requirements(self, fgraph):
        from aesara.graph.destroyhandler import DestroyHandler

        fgraph.attach_feature(DestroyHandler())

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, prof["opt"].op, file=stream)
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

    def candidate_input_idxs(self, node):
        if isinstance(node.op.scalar_op, aes.Composite) and len(node.outputs) > 1:
            # TODO: Implement specialized InplaceCompositeOptimizer with logic
            #  needed to correctly assign inplace for multi-output Composites
            return []
        else:
            return range(len(node.outputs))

    def apply(self, fgraph):
        r"""

        Attempts to replace all `Elemwise`\s by versions of them that operate
        inplace. It operates greedily: for each `Elemwise` that is encountered,
        for each output, it tries each input to see if it can operate inplace
        on that input. If so, it makes the change and goes to the next output
        or `Elemwise`.

        Examples
        --------

            x + y + z -> x += y += z
            (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)

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
        # The next longest rewriter is the canonizer phase.
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
                    i for i in self.candidate_input_idxs(node) if i not in baseline
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
                candidate_outputs = self.candidate_input_idxs(node)
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
                                fgraph.destroy_handler.root_destroyer.get(up_inp, None)
                                is inp.owner
                                for up_inp in updated_inputs
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
                                    "Some inplace rewriting was not "
                                    "performed due to an unexpected error:"
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
                            "Some inplace rewriting was not "
                            "performed due to an unexpected error"
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
compile.optdb.register(  # type: ignore
    "inplace_elemwise_opt",
    inplace_elemwise_optimizer,
    "inplace_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    position=75,
)


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
@node_rewriter([DimShuffle])
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
@node_rewriter([DimShuffle])
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
@node_rewriter([Elemwise])
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


def elemwise_max_input_fct(node):
    # `Elemwise.perform` uses NumPy ufuncs and they are limited to 31 inputs.
    if not config.cxx:
        return 31
    return 1024


class FusionOptimizer(GraphRewriter):
    """Graph optimizer that fuses consecutive Elemwise operations."""

    def __init__(self, local_optimizer=None):
        # TODO: Figure out what to do with this
        super().__init__()
        self.optimizer = local_optimizer

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    @staticmethod
    def elemwise_to_scalar(inputs, outputs):
        replace_inputs = [(inp, inp.clone()) for inp in inputs]
        outputs = clone_replace(outputs, replace=replace_inputs)
        # print("elemwise_to_scalar replaced outputs:")
        # aesara.dprint(outputs, print_type=True)

        inputs = [inp for _, inp in replace_inputs]
        fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=False)
        middle_inputs = []

        scalar_inputs = [
            aes.get_scalar_type(inp.type.dtype).make_variable() for inp in inputs
        ]
        middle_scalar_inputs = []

        for node in fg.toposort():
            node_scalar_inputs = []
            for inp in node.inputs:
                if inp in inputs:
                    node_scalar_inputs.append(scalar_inputs[inputs.index(inp)])
                elif inp in middle_inputs:
                    node_scalar_inputs.append(
                        middle_scalar_inputs[middle_inputs.index(inp)]
                    )
                else:
                    new_scalar_input = aes.get_scalar_type(
                        inp.type.dtype
                    ).make_variable()
                    node_scalar_inputs.append(new_scalar_input)
                    middle_scalar_inputs.append(new_scalar_input)
                    middle_inputs.append(inp)

            new_scalar_node = node.op.scalar_op.make_node(*node_scalar_inputs)
            middle_scalar_inputs.append(new_scalar_node.outputs[0])
            middle_inputs.append(node.outputs[0])

        scalar_outputs = [
            middle_scalar_inputs[middle_inputs.index(out)] for out in fg.outputs
        ]
        return scalar_inputs, scalar_outputs

    def apply(self, fgraph):
        nb_replacement = 0

        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        max_inputs = elemwise_max_input_fct(None)

        def find_next_fuseable_subgraph(
            fg: FunctionGraph,
        ) -> Generator[Tuple[List[Variable], List[Variable]], None, None]:
            """Find all subgraphs in a FunctionGraph that can be fused together

            Yields
            -------
            List of inputs and outputs that determine subgraphs which can be fused. This
            method assumes that such replacement is done across iterations of the
            generator.
            """

            @lru_cache(maxsize=None)
            def elemwise_scalar_op_has_c_code(node: Apply) -> bool:
                if node.op.scalar_op.supports_c_code(node.inputs, node.outputs):
                    return True
                else:
                    warn(
                        (
                            "Optimization Warning: "
                            f"The Op {node.op.scalar_op} does not provide a C implementation."
                            " As well as being potentially slow, this also disables "
                            "loop fusion."
                        )
                    )
                    return False

            # We start by creating two maps, 1) from each node to each potentially
            # fuseable client (both nodes must be single output Elemwise with same
            # broadcast type) and 2) from each node to each certainly unfuseable
            # client (those that don't fit into 1))
            fuseable_clients: Dict[Any, List[Any]] = defaultdict(list)
            unfuseable_clients: Dict[Any, List[Any]] = defaultdict(list)
            for out, clients in fg.clients.items():
                out_maybe_fuseable = (
                    out.owner
                    and isinstance(out.owner.op, Elemwise)
                    # and not isinstance(out.owner.op.scalar_op, aes.Composite)
                    and len(out.owner.outputs) == 1
                    and elemwise_scalar_op_has_c_code(out.owner)
                )
                for client, _ in clients:
                    if (
                        out_maybe_fuseable
                        and not isinstance(client, str)  # "output"
                        and isinstance(client.op, Elemwise)
                        # and not isinstance(client.op.scalar_op, aes.Composite)
                        and len(client.outputs) == 1
                        and out.type.broadcastable
                        == client.outputs[0].type.broadcastable
                        and elemwise_scalar_op_has_c_code(client)
                    ):
                        if client not in fuseable_clients[out]:
                            fuseable_clients[out].append(client)
                    else:
                        if client not in unfuseable_clients[out]:
                            unfuseable_clients[out].append(client)

            visited_nodes = set()
            while True:
                # print(
                #     "fuseable_clients:",
                #     {
                #         k: [out for v_ in v for out in v_.outputs]
                #         for k, v in fuseable_clients.items()
                #     },
                # )
                # print(
                #     "unfuseable_clients:",
                #     {
                #         k: [out for v_ in v if v_ != "output" for out in v_.outputs]
                #         for k, v in unfuseable_clients.items()
                #     },
                # )

                # We walk through the apply nodes looking for one that has at least one
                # candidate fuseable client
                toposort = fg.toposort()
                starting_nodes = set(toposort)
                for starting_node in toposort:
                    if starting_node in visited_nodes:
                        continue

                    starting_out = starting_node.outputs[0]
                    if not fuseable_clients.get(starting_out):
                        # print(f"\n> Skipping {out} as it has no fuseable clients")
                        visited_nodes.add(starting_node)
                        continue

                    subgraph_inputs: List[Variable] = []
                    subgraph_outputs: List[Variable] = []
                    unfuseable_clients_subgraph = set()
                    # Manually "deepcopy" clients mapping as those will be altered in place.
                    # Cannot use `copy.deepcopy` because that would also copy the Aesara variables.
                    fuseable_clients_temp: Dict[Any, List[Any]] = defaultdict(list)
                    unfuseable_clients_temp: Dict[Any, List[Any]] = defaultdict(list)
                    fuseable_clients_temp.update(
                        {
                            out: [client for client in clients]
                            for out, clients in fuseable_clients.items()
                        }
                    )
                    unfuseable_clients_temp.update(
                        {
                            out: [client for client in clients]
                            for out, clients in unfuseable_clients.items()
                        }
                    )
                    fuseable_nodes_to_visit = deque([starting_node])
                    # We now try to expand as much as possible towards the potentially
                    # fuseable clients and ancestors to detect the largest possible
                    # subgraph that can be Composed together into a single `Op`. The
                    # largest issue to watch out is for cyclical dependencies, where
                    # some inputs or clients may depend on other nodes of the same
                    # subgraph via a path that cannot be included in the Composite
                    # (unfuseable)
                    while fuseable_nodes_to_visit:
                        next_node = fuseable_nodes_to_visit.popleft()
                        visited_nodes.add(next_node)
                        next_out = next_node.outputs[0]
                        # print(f"\t{next_out=}, {subgraph_inputs=}, {subgraph_outputs=}, {fuseable_nodes_to_visit=}")

                        # Node must become an output if it is to be fused.
                        must_become_output = (
                            next_out not in fuseable_clients_temp
                            or next_out in unfuseable_clients_temp
                        )

                        # We have backtracked to this node, and it may no longer be a
                        # viable output
                        if must_become_output and next_out in subgraph_outputs:
                            subgraph_outputs.remove(next_out)
                            # unfuseable_clients_subgraph = (
                            #     unfuseable_clients_subgraph
                            #     - get_unfuseable_clients(unfuseable_clients_temp, out)
                            # )

                        required_unfuseable_inputs = [
                            inp
                            for inp in next_node.inputs
                            if next_node in unfuseable_clients_temp.get(inp, [])
                        ]

                        new_required_unfuseable_inputs = [
                            inp
                            for inp in required_unfuseable_inputs
                            if inp not in subgraph_inputs
                        ]

                        # print(f"\t\t{new_required_unfuseable_inputs=}, {required_unfuseable_inputs=}, {unfuseable_clients_subgraph=}")
                        must_backtrack = False
                        if new_required_unfuseable_inputs and subgraph_outputs:
                            # We need to check that any new ancestors required by this node
                            # do not depend on other outputs of the same subgraph, via
                            # an unfuseable path.
                            if any(
                                a in unfuseable_clients_subgraph
                                for a in ancestors(
                                    [next_out], blockers=subgraph_outputs
                                )
                            ):
                                # print("\t > Cannot fuse due to non-fuseable ancestor dependency in same subgraph")
                                must_backtrack = True

                        if not must_backtrack:
                            implied_unfuseable_clients = {
                                c
                                for client in unfuseable_clients_temp.get(next_out, [])
                                if client != "output"
                                for c in client.outputs
                            }

                            new_implied_unfuseable_clients = [
                                client
                                for client in implied_unfuseable_clients
                                if client not in unfuseable_clients_subgraph
                            ]

                            if new_implied_unfuseable_clients and subgraph_inputs:
                                # We need to check that any ancestors of the subgraph do not depend
                                # on other clients of this node, via an unfuseable path.
                                if any(
                                    a in new_implied_unfuseable_clients
                                    for a in ancestors(subgraph_inputs)
                                ):
                                    # print("\t > Cannot fuse due to non-fuseable client dependency in same subgraph")
                                    must_backtrack = True

                        if must_backtrack:
                            for inp in next_node.inputs:
                                if (
                                    inp.owner in visited_nodes
                                    # next_node could have the same input repeated
                                    and next_node in fuseable_clients_temp[inp]
                                ):
                                    fuseable_clients_temp[inp].remove(next_node)
                                    unfuseable_clients_temp[inp].append(next_node)
                                    # print(f"\t\t: Will have to revisit {inp} as it must now become an output of subgraph")
                                    fuseable_nodes_to_visit.appendleft(inp.owner)

                            for client in fuseable_clients_temp[next_out]:
                                if client in visited_nodes:
                                    # MyPy does not know that fuseable clients can never be `output` clients
                                    client = typing_cast(Apply, client)
                                    fuseable_clients_temp[next_out].remove(client)
                                    unfuseable_clients_temp[next_out].append(client)
                                    # print(f"\t\t: Will have to revisit {client} as current node must now become an input of subgraph")
                                    fuseable_nodes_to_visit.appendleft(client)

                            # Revisit node at a later time
                            visited_nodes.remove(next_node)
                            continue

                        for inp in new_required_unfuseable_inputs:
                            # Node could require the same new input multiple times
                            if inp not in subgraph_inputs:
                                subgraph_inputs.append(inp)

                        if must_become_output:
                            # print("\t\tMust become output!")
                            subgraph_outputs.append(next_out)
                            # This node is now a "definite" part of the fused graph
                            unfuseable_clients_subgraph.update(
                                new_implied_unfuseable_clients
                            )

                        for inp in sorted(
                            (
                                inp
                                for inp in next_node.inputs
                                if (
                                    inp not in required_unfuseable_inputs
                                    # No need to check if inp.owner is not None, as that
                                    # would by definition be a required_unfuseable_input
                                    and inp.owner not in visited_nodes
                                )
                            ),
                            key=lambda inp: toposort.index(inp.owner),
                            reverse=True,
                        ):
                            # Expand through unvisited fuseable ancestors
                            fuseable_nodes_to_visit.appendleft(inp.owner)

                        for next_node in sorted(
                            fuseable_clients_temp.get(next_out, []),
                            key=lambda node: toposort.index(node),
                        ):
                            # Expand through unvisited fuseable clients
                            if next_node not in visited_nodes:
                                fuseable_nodes_to_visit.append(next_node)

                    # print(f"\t~ final fused subgraph: {subgraph_inputs=}, {subgraph_outputs=}")

                    # Don't yield if final subgraph is just the original Elemwise
                    if (
                        len(subgraph_outputs) == 1
                        and (
                            len(subgraph_outputs[0].owner.inputs)
                            == len(subgraph_inputs)
                        )
                        and (
                            set(subgraph_outputs[0].owner.inputs)
                            == set(subgraph_inputs)
                        )
                    ):
                        # print(f"\t! final fused subgraph is just the original elemwise")
                        # Update fuseable mappings
                        # No input was actually fuseable
                        for inp in starting_node.inputs:
                            if (
                                inp in fuseable_clients
                                and starting_node in fuseable_clients[inp]
                            ):
                                fuseable_clients[inp].remove(starting_node)
                                unfuseable_clients[inp].append(starting_node)
                        # No client was actually fuseable
                        for client in fuseable_clients.pop(starting_out, []):
                            unfuseable_clients[starting_out].append(client)

                    else:
                        yield subgraph_inputs, subgraph_outputs

                        # This is where we avoid repeated work by using a stateful
                        # generator. For large models (as in `TestFusion.test_big_fusion`)
                        # this can provide huge speedups

                        # Update fuseable mappings
                        next_nodes = fg.apply_nodes
                        (new_composite_node,) = next_nodes - starting_nodes
                        dropped_nodes = starting_nodes - next_nodes

                        # Remove intermediate Composite nodes from mappings
                        for dropped_node in dropped_nodes:
                            (dropped_out,) = dropped_node.outputs
                            fuseable_clients.pop(dropped_out, None)
                            unfuseable_clients.pop(dropped_out, None)
                            visited_nodes.remove(dropped_node)

                        # Any input is now definitely unfuseable
                        for inp in subgraph_inputs:
                            if inp in fuseable_clients:
                                new_fuseable_clients = [
                                    client
                                    for client in fuseable_clients[inp]
                                    if client not in dropped_nodes
                                ]
                                if new_fuseable_clients:
                                    fuseable_clients[inp] = new_fuseable_clients
                                else:
                                    fuseable_clients.pop(inp)
                            unfuseable_clients[inp] = [
                                client
                                for client in unfuseable_clients[inp]
                                if client not in dropped_nodes
                            ] + [new_composite_node]

                        # Any client is now definitely unfuseable
                        for out in new_composite_node.outputs:
                            unfuseable_clients[out] = [
                                client for client, _ in fg.clients[out]
                            ]
                        visited_nodes.add(new_composite_node)
                    break
                else:  # nobreak
                    return

        # aesara.dprint(fgraph, print_type=True)
        for res in find_next_fuseable_subgraph(fgraph):
            # print(f">> >> Start of iteration {nb_replacement}: {len(fgraph.apply_nodes)=}")
            if res is None:
                # print("<< No further fuseable subgraph found")
                break
            inputs, outputs = res

            if len(inputs) > max_inputs:
                warn(
                    "Loop fusion failed because the resulting node would exceed "
                    "the kernel argument limit."
                )
                break

            scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
            composite_outputs = Elemwise(aes.Composite(scalar_inputs, scalar_outputs))(
                *inputs
            )
            if not isinstance(composite_outputs, list):
                composite_outputs = [composite_outputs]
            for old_out, composite_out in zip(outputs, composite_outputs):
                if old_out.name:
                    composite_out.name = old_out.name

            # print(f"{outputs=},\n{composite_outputs=},\n{inputs=}")
            fgraph.replace_all_validate(
                list(zip(outputs, composite_outputs)),
                reason=self.__class__.__name__,
            )
            # print(f"<< <<End of iteration {nb_replacement} {len(fgraph.apply_nodes)=}\n")
            # aesara.dprint(fgraph, print_type=True)
            nb_replacement += 1

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
            1,  # nb_iter
            nb_replacement,
            0,  # nb_inconsintency_replace
            validate_time,
            callback_time,
            callbacks_time,
            0,  # toposort_time
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        # TODO: Update this
        blanc = "    " * level
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] is not None and prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)
        print(blanc, " time_toposort", prof[7], file=stream)


fuse_seqopt = SequenceDB()
if config.tensor__local_elemwise_fusion:
    fuse_seqopt.register(
        "composite_elemwise_fusion",
        FusionOptimizer(),
        "fast_run",
        "fusion",
        position=1,
    )
    # Position before AddDestroyHandler(49.5)
    compile.optdb.register(  # type: ignore
        "elemwise_fusion",
        fuse_seqopt,
        "fast_run",
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
        position=49,
    )
else:
    compile.optdb.register(  # type: ignore
        "elemwise_fusion",
        FusionOptimizer(),
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
        position=49,
    )


@register_canonicalize
@node_rewriter([Elemwise])
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

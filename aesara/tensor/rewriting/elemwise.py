import sys
import time
from collections import defaultdict
from typing import Optional
from warnings import warn

import aesara
import aesara.scalar.basic as aes
from aesara import compile
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, io_toposort
from aesara.graph.features import ReplaceValidate
from aesara.graph.op import compute_test_value, get_test_value
from aesara.graph.rewriting.basic import GraphRewriter, copy_stack_trace, node_rewriter
from aesara.graph.rewriting.db import SequenceDB
from aesara.graph.utils import InconsistencyError, MethodNotDefined, TestValueError
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


def local_elemwise_fusion_op(op_class, max_input_fct=lambda node: 32, maker=None):
    r"""Create a recursive function that fuses `Elemwise` `Op`\s.

    The basic idea is that we loop through an `Elemwise` node's inputs, find
    other `Elemwise` nodes, determine the scalars input types for all of the
    `Elemwise` `Op`\s, construct a new scalar `Op` using the scalar input types
    and each `Elemwise`'s scalar `Op`, and use the composite scalar `Op` in a
    new "fused" `Elemwise`.

    It's parameterized in order to work for `Elemwise` `Op`\s.

    Parameters
    ----------
    op_class : type
        `Elemwise` class (the one that we want to fuse)
    max_input_fct : callable
        A function that returns the maximum number of inputs that this `Elemwise`
        can take.
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
        # TODO: use broadcast flag?

        # TODO: don't do this rewrite as a `NodeRewriter`.
        # Analyze the graph in terms of elemwise subgraphs, and then
        # replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't
        # fit within the parameter space of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs
        # have multiple clients. This can't be done with a `NodeRewriter`

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
            scalar_node: Optional[Apply] = None
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
                                # Sometimes the original inputs have
                                # zero-valued shapes in some dimensions, which
                                # implies that this whole scalar thing doesn't
                                # make sense (i.e. we're asking for the scalar
                                # value of an entry in a zero-dimensional
                                # array).
                                # This will eventually lead to an error in the
                                # `compute_test_value` call below when/if
                                # `config.compute_test_value_opt` is enabled
                                # (for debugging, more or less)
                                tmp.tag.test_value = tv.item()
                            except (TestValueError, ValueError):
                                pass

                            tmp_s_input.append(tmp)
                            tmp_input.append(ii)
                            tmp_scalar.append(tmp_s_input[-1])

                    # Use the `Op.make_node` interface in case `Op.__call__`
                    # has been customized
                    scalar_node = i.owner.op.scalar_op.make_node(*tmp_s_input)

                    if config.compute_test_value_opt != "off":
                        # This is required because `Op.make_node` won't do it
                        compute_test_value(scalar_node)

                    # If the scalar_op doesn't have a C implementation, we skip
                    # its fusion to allow fusion of the other ops
                    i.owner.op.scalar_op.c_code(
                        scalar_node,
                        "test_presence_of_c_code",
                        ["x" for x in i.owner.inputs],
                        ["z" for z in i.owner.outputs],
                        {"fail": "%(fail)s"},
                    )

                except (NotImplementedError, MethodNotDefined):
                    warn(
                        (
                            "Rewrite warning: "
                            f"The Op {i.owner.op.scalar_op} does not provide a C implementation."
                            " As well as being potentially slow, this also disables "
                            "loop fusion."
                        )
                    )
                    scalar_node = None

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

            if scalar_node and (new_nb_input_ <= max_nb_input):
                fused = True
                new_nb_input = new_nb_input_
                inputs.extend(tmp_input)
                s_inputs.extend(tmp_scalar)
                s_g.extend(scalar_node.outputs)
            else:
                # We must support the case where the same variable appears many
                # times within the inputs
                if inputs.count(i) == node.inputs.count(i):
                    s = s_inputs[inputs.index(i)]
                else:
                    s = aes.get_scalar_type(i.type.dtype).make_variable()
                    if config.compute_test_value_opt != "off":
                        try:
                            v = get_test_value(i)
                            # See the zero-dimensional test value situation
                            # described above.
                            s.tag.test_value = v.item()
                        except (TestValueError, ValueError):
                            pass

                    inputs.append(i)
                    s_inputs.append(s)
                s_g.append(s)

        if not fused:
            return False

        if new_nb_input != len(inputs) or len(s_inputs) != len(inputs):
            # TODO FIXME: This shouldn't be a generic `Exception`
            raise Exception(
                "Something has gone wrong with the elemwise fusion rewrite; skipping."
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
            warn(
                (
                    "Rewrite warning: "
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
            warn(
                "Loop fusion failed because the resulting node "
                "would exceed the kernel argument limit."
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


class FusionOptimizer(GraphRewriter):
    """Graph rewriter that simply runs node fusion operations.

    TODO: This is basically an `EquilibriumGraphRewriter`; we should just use that.

    """

    def __init__(self, node_rewriter):
        super().__init__()
        self.node_rewriter = node_rewriter

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

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
                    new_outputs = self.node_rewriter(fgraph, node)
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

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
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


if config.tensor__local_elemwise_fusion:
    # Must be after gpu(48.5) and before AddDestroyHandler(49.5)
    fuse_seqopt = SequenceDB()
    fuse_seqopt.register(
        "composite_elemwise_fusion",
        FusionOptimizer(local_elemwise_fusion),
        "fast_run",
        "fusion",
        position=1,
    )
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
        FusionOptimizer(local_elemwise_fusion),
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

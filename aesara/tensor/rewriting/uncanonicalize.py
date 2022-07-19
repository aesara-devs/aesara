"""
This file implement specialization optimization that break the
canonization form of the graph.

Currently there is problem with the order of optimization and the
definition of definition of canonized graph.

Right now there is a canonization optimization phase that try to make
all equivalent graph identical. This is not always the case, but it do
many of the basic stuff canonical. We need to extend the definition of
canonization to make this true more often.

The problem this file indent to fix in the future is that in the
"Equilibrium" specialization optimization phase, there is optimization
that request that the graph is canonical, some other request that this
is not true, and some other that break the canonicalization for some
optimization. As we can't control the order of those optimization, there
is case that some optimization requesting a canonical graph won't be
applied as optimization that break the canonicalization form of the
graph executed before.

To fix this, we need to split the specialization phase into a phase
where optimization can't break the canonicalization form and one where
this is allowed. This is also needed for the stabilized optimization
phase, but as it happen before the specialization phase, this cause less
problem.

Also, we should make the fgraph refuse optimization that break the
canonization of the graph in the optimizations phases where the graph is
supposed to be canonical.

"""

from aesara import scalar as aes
from aesara.graph.rewriting.basic import copy_stack_trace, node_rewriter
from aesara.tensor.basic import Alloc, alloc, constant
from aesara.tensor.elemwise import CAReduce, DimShuffle
from aesara.tensor.math import Argmax, Max, MaxAndArgmax, Min, neg
from aesara.tensor.rewriting.basic import register_uncanonicalize
from aesara.tensor.shape import Reshape, reshape
from aesara.tensor.subtensor import Subtensor


@register_uncanonicalize
@node_rewriter([MaxAndArgmax])
def local_max_and_argmax(fgraph, node):
    """
    If we don't use the argmax, change it to a max only.
    """
    if isinstance(node.op, MaxAndArgmax):
        axis = node.op.get_params(node)
        if len(fgraph.clients[node.outputs[1]]) == 0:
            new = Max(axis)(node.inputs[0])
            copy_stack_trace(node.outputs[0], new)
            return [new, None]

        if len(fgraph.clients[node.outputs[0]]) == 0:
            new = Argmax(axis)(node.inputs[0])
            copy_stack_trace(node.outputs[0], new)
            return [None, new]


@register_uncanonicalize
@node_rewriter([neg])
def local_max_to_min(fgraph, node):
    """
    Change -(max(-x)) to min.

    This is tested in tensor/tests/test_basic.py:test_min_max.

    Notes
    -----
    We don't need an opt that will do the reverse as by default
    the interface put only MaxAndArgmax into the graph.

    """
    if node.op == neg and node.inputs[0].owner:
        max = node.inputs[0]
        if (
            max.owner
            and isinstance(max.owner.op, CAReduce)
            and max.owner.op.scalar_op == aes.scalar_maximum
        ):
            neg_node = max.owner.inputs[0]
            if neg_node.owner and neg_node.owner.op == neg:
                new = Min(max.owner.op.axis)(neg_node.owner.inputs[0])
                return [copy_stack_trace(node.outputs[0], new)]

    return False


@register_uncanonicalize
@node_rewriter([Alloc])
def local_alloc_dimshuffle(fgraph, node):
    """
    If a dimshuffle is inside an alloc and only adds dimension to the
    left, remove it.

    Alloc(DimShuffle(x), ...) - > Alloc(x, ...)
    """
    if isinstance(node.op, Alloc):
        input_ = node.inputs[0]
        if input_.owner and isinstance(input_.owner.op, DimShuffle):
            # check if it only adds dimension to the left
            new_order = input_.owner.op.new_order
            expected_new_order = ("x",) * (
                input_.ndim - input_.owner.inputs[0].ndim
            ) + tuple(range(input_.owner.inputs[0].ndim))
            if new_order != expected_new_order:
                return False
            return [alloc(input_.owner.inputs[0], *node.inputs[1:])]
    return False


@register_uncanonicalize
@node_rewriter([Reshape])
def local_reshape_dimshuffle(fgraph, node):
    """
    If a dimshuffle is inside a reshape and does not change the order
    of dimensions, remove it.

    Reshape(Dimshuffle(x), shp) -> Reshape(x, shp)
    """
    if isinstance(node.op, Reshape):
        input_ = node.inputs[0]
        if input_.owner and isinstance(input_.owner.op, DimShuffle):
            new_order = input_.owner.op.new_order
            offset = 0
            for dim in new_order:
                if dim == "x":
                    continue
                elif dim != offset:
                    return False
                else:
                    offset += 1
            return [
                reshape(
                    input_.owner.inputs[0], node.inputs[1], ndim=node.outputs[0].ndim
                )
            ]
    return False


@register_uncanonicalize
@node_rewriter([DimShuffle])
def local_dimshuffle_alloc(fgraph, node):
    """
    If an alloc is inside a dimshuffle which only adds dimension to the left,
    scrap the dimshuffle and adds 1 into the alloc

    dimshuffle{x, 0, 1}(alloc([3 4], 3, 2) => alloc([3 4], 1, 3, 2)
    """
    if isinstance(node.op, DimShuffle) and node.inputs[0].owner:
        input_ = node.inputs[0]
        if isinstance(input_.owner.op, Alloc):
            # check if it only adds dimension to the left
            new_order = node.op.new_order
            expected_new_order = ("x",) * (len(new_order) - input_.ndim) + tuple(
                range(input_.ndim)
            )
            if new_order != expected_new_order:
                return False

            # count numbers of 'x'
            nb_new_dims = len(new_order) - input_.ndim
            new_shape_input = (1,) * nb_new_dims + tuple(input_.owner.inputs[1:])

            return [alloc(input_.owner.inputs[0], *new_shape_input)]
    return False


@register_uncanonicalize
@node_rewriter([DimShuffle])
def local_dimshuffle_subtensor(fgraph, node):
    """If a subtensor is inside a dimshuffle which only drop
    broadcastable dimensions, scrap the dimshuffle and index the
    subtensor with 0

    x[i:j, :, k:l].dimshuffle(0, 2) =>
        x[i:j, 0, k:l] if x.broadcastable == (False, True, False)

    """
    if isinstance(node.op, DimShuffle) and node.inputs[0].owner:
        # the dimshuffle can only drop dimensions (cannot reshape nor add 'x')
        if "x" in node.op.new_order:
            return False
        new_order = node.op.new_order
        # new order could be empty
        # Verif that we don't change dimensions order.
        if len(new_order) > 1:
            past_dim = new_order[0]
            for dim in new_order[1:]:
                if not dim > past_dim:
                    return False
                else:
                    past_dim = dim

        input_ = node.inputs[0]
        if isinstance(input_.owner.op, Subtensor):
            # the arguments missing from the dimshuffles must be dims
            # that are broadcastable
            broadcastable = input_.broadcastable

            missing_dims = list(range(input_.ndim))
            for dim in new_order:
                missing_dims.remove(dim)

            if not all(broadcastable[i] for i in missing_dims):
                return False

            # create a new idx_list for a new Subtensor object
            # have to loop on idx_list and inputs
            # inputs has the length of sum of non None elements of idx_list
            # (check in slice!).
            # len(missing_dims) can be < len(idx_list), this happens if
            # tensor was indexed such as x[scalar, :, :], check that as well
            new_idx_list = list(input_.owner.op.idx_list)
            new_inputs = [input_.owner.inputs[0]]
            zero = constant(0)
            slice_attr_list = ["start", "stop", "step"]
            j = 0
            slice_i = -1
            subtensor_removed_dims = 0
            for i, idx in enumerate(input_.owner.op.idx_list):
                if isinstance(idx, slice):
                    past_j = j
                    slice_i += 1
                    for slice_attr in slice_attr_list:
                        if getattr(idx, slice_attr) is not None:
                            new_inputs += [input_.owner.inputs[1 + j]]
                            j += 1
                    # if past_j == j indicates a slice(None, None, None),
                    # that's where we want to index with 0 if it is also at
                    # the same spot of a missing dim
                    if past_j == j and slice_i in missing_dims:
                        new_idx_list[i] = zero
                        new_inputs += [zero]
                else:
                    new_inputs += [input_.owner.inputs[1 + j]]
                    j += 1
                    subtensor_removed_dims += 1
            # Verify the trailing dimensions the subtensor didn't look at.
            for idx in range(len(input_.owner.op.idx_list), new_inputs[0].ndim):
                if (idx - subtensor_removed_dims) in missing_dims:
                    while len(new_idx_list) < idx:
                        new_idx_list.append(slice(None))

                    new_idx_list.append(zero)
                    new_inputs.append(zero)
            return [Subtensor(new_idx_list)(*new_inputs)]
    return False

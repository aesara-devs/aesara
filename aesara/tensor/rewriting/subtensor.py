import sys
from collections.abc import Iterable

import numpy as np

import aesara
import aesara.scalar.basic as aes
from aesara import compile
from aesara.graph.basic import Constant, Variable
from aesara.graph.rewriting.basic import (
    WalkingGraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from aesara.raise_op import Assert
from aesara.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
    alloc,
    as_tensor,
    cast,
    concatenate,
    extract_constant,
    get_scalar_constant_value,
    switch,
)
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import Dot, add
from aesara.tensor.math import all as at_all
from aesara.tensor.math import (
    and_,
    ceil_intdiv,
    dot,
    eq,
    ge,
    gt,
    le,
    lt,
    maximum,
    minimum,
    or_,
)
from aesara.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from aesara.tensor.shape import (
    Shape,
    SpecifyShape,
    Unbroadcast,
    shape_padleft,
    shape_tuple,
    specify_shape,
    unbroadcast,
)
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor1,
    advanced_subtensor,
    advanced_subtensor1,
    as_index_constant,
    as_index_literal,
    get_canonical_form_slice,
    get_constant_idx,
    get_idx_list,
    get_slice_elements,
    inc_subtensor,
    indices_from_subtensor,
)
from aesara.tensor.type import TensorType
from aesara.tensor.type_other import NoneTypeT, SliceConstant, SliceType
from aesara.tensor.var import TensorConstant, TensorVariable


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


def transform_take(a, indices, axis):
    r"""Transform ``arr[:,:,:,indices,...]``-like operations into single-dimensional, vector index operations.

    This effectively converts certain `AdvancedSubtensor` `Op`\s into a
    combination of `AdvancedSubtensor1`, `Dimshuffle`, and `Reshape` `Op`\s,
    which can be more efficient.

    Parameters
    ----------
    a : TensorVariable
        The source array.
    indices : TensorVariable, ndarray, list, tuple
        The indices of the values to extract.
    axis : int
        The axis over which to select values. By default, the flattened
        input array is used.

    """
    a = aesara.tensor.as_tensor_variable(a)
    indices = aesara.tensor.as_tensor_variable(indices)
    # We can use the more efficient `AdvancedSubtensor1` if `indices` is a vector
    if indices.ndim == 1:
        if axis == 0:
            return advanced_subtensor1(a, indices)
        else:
            shuffle = list(range(a.ndim))
            shuffle[0] = axis
            shuffle[axis] = 0
            res = advanced_subtensor1(a.dimshuffle(shuffle), indices).dimshuffle(
                shuffle
            )
            return res

    # We can reshape and flatten the indices in order to use an
    # `AdvancedSubtensor1` `Op` per the above
    indices_shape = shape_tuple(indices)
    a_shape = shape_tuple(a)

    shape_parts = [
        a_shape[:axis],
        indices_shape,
        a_shape[axis + 1 :],
    ]

    shape_parts = [sp for sp in shape_parts if len(sp) > 0]

    assert len(shape_parts) > 0

    if len(shape_parts) > 1:
        shape = aesara.tensor.concatenate(shape_parts)
    else:
        shape = shape_parts[0]

    ndim = a.ndim + indices.ndim - 1

    return transform_take(a, indices.flatten(), axis).reshape(shape, ndim)


def is_full_slice(x):
    """Determine if `x` is a ``slice(None)`` or a symbolic equivalent."""
    if (
        (isinstance(x, slice) and x == slice(None))
        or (isinstance(x, SliceConstant) and x.value == slice(None))
        or (
            not isinstance(x, SliceConstant)
            and isinstance(getattr(x, "type", None), SliceType)
            and x.owner is not None
            and all(
                isinstance(getattr(i, "type", None), NoneTypeT) for i in x.owner.inputs
            )
        )
    ):
        return True
    return False


def get_advsubtensor_axis(indices):
    """Determine the axis at which an array index is applied.

    This only works for ``take``-like indices: e.g. ``x[:, :, idx, ...]``.  For
    the above example, `get_advsubtensor_axis` would return ``2``.  If it
    encounters anything other than a set of `indices` containing full slices
    and an array/tensor index, it will return ``None``.

    """
    found_idx = False
    axis = 0
    for idx in indices:
        if not found_idx and is_full_slice(idx):
            # Preceding full slices
            axis += 1
        elif found_idx and not is_full_slice(idx):
            # We don't handle multiple indices
            return
        elif found_idx and is_full_slice(idx):
            # Trailing full slices
            continue
        else:
            found_idx = True

    if isinstance(
        indices[axis], (TensorConstant, TensorVariable, TensorSharedVariable)
    ):
        return axis


@register_specialize
@node_rewriter([AdvancedSubtensor])
def local_replace_AdvancedSubtensor(fgraph, node):
    r"""
    This rewrite converts expressions like ``X[..., y]`` into ``X.T[y].T``, for
    a vector ``y``, and ``X[z, ...]`` into ``X[z.flatten()].reshape(...)``, for a
    matrix ``z``.

    These rewrites replace `AdvancedSubtensor`\s with the more efficient
    `AdvancedSubtensor1` and `Subtensor` `Op`\s.
    """

    if not isinstance(node.op, AdvancedSubtensor):
        return

    indexed_var = node.inputs[0]
    indices = node.inputs[1:]

    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't handled
        return

    new_res = transform_take(indexed_var, indices[axis], axis)
    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_specialize
@node_rewriter([AdvancedIncSubtensor])
def local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1(fgraph, node):
    r"""Replace `AdvancedIncSubtensor`\s with `AdvancedIncSubtensor1`\s.

    This is only done when there's a single vector index.
    """

    if not isinstance(node.op, AdvancedIncSubtensor) or node.op.ignore_duplicates:
        # `AdvancedIncSubtensor1` does not ignore duplicate index values
        return

    res = node.inputs[0]
    val = node.inputs[1]
    indices = node.inputs[2:]

    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't currently handled by `AdvancedIncSubtensor1`
        return

    new_subtensor = transform_take(res, indices[axis], axis)

    new_res = inc_subtensor(
        new_subtensor,
        val,
        inplace=node.op.inplace,
        set_instead_of_inc=node.op.set_instead_of_inc,
        ignore_duplicates=False,
    )
    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_dot(fgraph, node):
    """Rewrite ``at.dot(A, B)[idxs]`` into ``at.dot(A[idxs_a], B[idxs_b])``.
    ``idxs_a`` is the first ``A.ndim-1`` entries of ``idxs``, and ``idxs_b`` is
    the remaining entries of ``idxs`` (if any), modified to skip the
    second-to-last dimension of ``B`` (because dot sums over this dimension).
    """
    if not isinstance(node.op, Subtensor):
        return
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Dot):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(fgraph.clients[node.inputs[0]]) > 1:
        return

    a = node.inputs[0].owner.inputs[0]
    b = node.inputs[0].owner.inputs[1]

    idx_list = get_idx_list(node.inputs, node.op.idx_list)

    num_a_indices = min(a.ndim - 1, len(idx_list))
    a_indices = idx_list[:num_a_indices]
    b_indices = idx_list[num_a_indices:]

    # This is necessary because np.dot sums the last index of a with the second to last of b
    # so we want to skip the second-to-last index into b.
    # This wasn't necessary for a, because we just omitted the last index.
    # We skip this if b.ndim = 1, since then we just want b_sub = b, not b_sub = b[:]
    # (dot also handles b.ndim < 2 as a special case)
    if b.ndim > 1 and len(b_indices) >= b.ndim - 1:
        b_indices = (
            b_indices[: b.ndim - 2]
            + (slice(None, None, None),)
            + b_indices[b.ndim - 2 :]
        )

    a_sub = a.__getitem__(tuple(a_indices))
    b_sub = b.__getitem__(tuple(b_indices)) if b_indices else b

    # Copy over previous output stacktrace to a_sub and b_sub,
    # because an error in the subtensor operation (e.g. an index error)
    # on either a or b must correspond to an error in the
    # subtensor operation on their dot product.
    copy_stack_trace(node.outputs[0], [a_sub, b_sub])

    # Copy over previous output stacktrace and previous dot product stacktrace,
    # because an error here may correspond to an either in either the original
    # dot product, or in the dot product after the subtensor operation.
    r = dot(a_sub, b_sub)
    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_useless_slice(fgraph, node):
    """
    Remove Subtensor of the form X[0, :] -> X[0]
    """
    if isinstance(node.op, Subtensor):
        slices = get_idx_list(node.inputs, node.op.idx_list)
        last_slice = len(slices)
        for s in slices[::-1]:
            # check if slice and then check slice indices
            if (
                isinstance(s, slice)
                and s.start is None
                and s.stop is None
                and (
                    s.step is None
                    or extract_constant(s.step, only_process_constants=True) == 1
                )
            ):
                last_slice -= 1
            else:
                break
        # check if we removed something
        if last_slice < len(slices):
            subtens = Subtensor(slices[:last_slice])
            sl_ins = get_slice_elements(
                slices[:last_slice], lambda x: isinstance(x, Variable)
            )
            out = subtens(node.inputs[0], *sl_ins)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, out)
            return [out]


# fast_compile to allow opt subtensor(cast{float32}(make_vector))
@register_canonicalize("fast_compile")
@node_rewriter([Subtensor])
def local_subtensor_lift(fgraph, node):
    """
    unary(x)[idx] -> unary(x[idx])#any broadcast pattern.

    Handles the following unary ops:
    elemwise(x,...)[idx] -> elemwise(x[idx],...)
      when x,... are broadcasted scalar or not broadcasted at all
    Unbroadcast(x)[idx] => Unbroadcast(x[idx])

    """
    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if not u.owner or len(fgraph.clients[u]) > 1:
            return False

        if isinstance(u.owner.op, Elemwise) and len(u.owner.inputs) == 1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, x_idx)
            ret = u.owner.op(x_idx)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
            return [ret]

        if isinstance(u.owner.op, Elemwise):
            new_inputs = []
            if all(sum(i.type.broadcastable) == 0 for i in u.owner.inputs):
                # There is no broadcastable in the inputs
                idx = node.inputs[1:]
                new_inputs = [node.op(i, *idx) for i in u.owner.inputs]
                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]
            elif all(sum(i.type.broadcastable) in [i.ndim, 0] for i in u.owner.inputs):
                # There is no broadcastable in the inputs or it is scalar
                idx = node.inputs[1:]
                new_inputs = []
                for i in u.owner.inputs:
                    if sum(i.type.broadcastable) == 0:
                        new_inputs.append(node.op(i, *idx))
                    else:
                        # If the subtensor remove some dims, we must
                        # lower the number of dimensions of this scalar.
                        if node.outputs[0].ndim == i.ndim:
                            new_inputs.append(i)
                        else:
                            new_inputs.append(
                                i.dimshuffle(["x"] * node.outputs[0].ndim)
                            )

                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]

        if isinstance(u.owner.op, Unbroadcast):
            # Subtensor might reduce dim., adapt broadcast pattern accordingly
            old_axes = u.owner.op.axes
            new_axes = []

            # loop through indices being subtensor-ed
            # i indexes broadcastable pattern before subtensor
            # j indexes broadcastable pattern after subtensor
            j = 0
            for (i, x) in enumerate(node.op.idx_list):
                # if it is not a slice, it will reduce the dimension, should
                # not appear in the broascastable dimensions
                if isinstance(x, slice):
                    if i in old_axes:
                        new_axes.append(j)
                    j += 1
            # now keep the broadcastable pattern of all
            # items not appearing in subtensor list
            for i in range(len(node.op.idx_list), len(u.broadcastable)):
                if i in old_axes:
                    new_axes.append(j)
                j += 1

            subt_x = node.op(u.owner.inputs[0], *node.inputs[1:])
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs[0], subt_x)

            rbcast_subt_x = unbroadcast(subt_x, *new_axes)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], rbcast_subt_x)

            return [rbcast_subt_x]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_merge(fgraph, node):
    """
    Refactored optimization to deal with all cases of tensor merging.
    Given a subgraph of the form Subtensor(Subtensor(u)), the optimization
    expresses all slices in a canonical form, and then merges them together.

    """

    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if u.owner and isinstance(u.owner.op, Subtensor):
            # We can merge :)
            # x actual tensor on which we are picking slices
            x = u.owner.inputs[0]
            # slices of the first applied subtensor
            slices1 = get_idx_list(u.owner.inputs, u.owner.op.idx_list)
            slices2 = get_idx_list(node.inputs, node.op.idx_list)
            # Get the shapes of the vectors !
            try:
                # try not to introduce new shape into the graph
                xshape = fgraph.shape_feature.shape_of[x]
                ushape = fgraph.shape_feature.shape_of[u]
            except AttributeError:
                # Following the suggested use of shape_feature which should
                # consider the case when the compilation mode doesn't
                # include the ShapeFeature
                xshape = x.shape
                ushape = u.shape

            merged_slices = []
            pos_2 = 0
            pos_1 = 0
            while (pos_1 < len(slices1)) and (pos_2 < len(slices2)):
                slice1 = slices1[pos_1]
                if isinstance(slice1, slice):
                    merged_slices.append(
                        merge_two_slices(
                            fgraph, slice1, xshape[pos_1], slices2[pos_2], ushape[pos_2]
                        )
                    )
                    pos_2 += 1
                else:
                    merged_slices.append(slice1)
                pos_1 += 1

            if pos_2 < len(slices2):
                merged_slices += slices2[pos_2:]
            else:
                merged_slices += slices1[pos_1:]

            merged_slices = tuple(as_index_constant(s) for s in merged_slices)
            subtens = Subtensor(merged_slices)

            sl_ins = get_slice_elements(
                merged_slices, lambda x: isinstance(x, Variable)
            )
            # Do not call make_node for test_value
            out = subtens(x, *sl_ins)

            # Copy over previous output stacktrace
            # and stacktrace from previous slicing operation.
            # Why? Because, the merged slicing operation could have failed
            # because of either of the two original slicing operations
            orig_out = node.outputs[0]
            copy_stack_trace([orig_out, node.inputs[0]], out)
            return [out]


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_remove_broadcastable_index(fgraph, node):
    """
    Remove broadcastable dimension with index 0 or -1
    a[:,:,:,0] -> a.dimshuffle(0,1,2), when
        a.broadcastable = (False, False, False, True)
    a[0,:,-1,:] -> a.dimshuffle(1,3), when
        a.broadcastable = (True, False, True, False)

    """
    if isinstance(node.op, Subtensor):
        idx = node.op.idx_list
    else:
        return

    remove_dim = []
    node_inputs_idx = 1
    for dim, elem in enumerate(idx):
        if isinstance(elem, (aes.ScalarType)):
            # The idx is a ScalarType, ie a Type. This means the actual index
            # is contained in node.inputs[1]
            dim_index = node.inputs[node_inputs_idx]
            if isinstance(dim_index, aes.ScalarConstant):
                dim_index = dim_index.value
            if dim_index in (0, -1) and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
                node_inputs_idx += 1
            else:
                return
        elif isinstance(elem, slice):
            if elem != slice(None):
                return
        elif isinstance(elem, (int, np.integer)):
            if elem in (0, -1) and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
        else:
            raise TypeError("case not expected")

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_alloc(fgraph, node):
    """

    alloc(val)[x:y] -> alloc(val[...])
    alloc(val)[x:y] -> alloc(val)
    This can be seen as a lift, but it also reduce the number of computation/memory.

    """
    if not isinstance(node.op, Subtensor):
        return False
    u = node.inputs[0]
    if u.owner is None:
        return False
    if not isinstance(u.owner.op, Alloc):
        return False
    slices = get_idx_list(node.inputs, node.op.idx_list)
    val = u.owner.inputs[0]
    dims = u.owner.inputs[1:]
    assert len(slices) <= len(dims)

    # Number of dimensions added to val
    n_added_dims = u.ndim - val.ndim
    # Dimensions of the returned alloc
    nw_dims = []
    # Slices to take from val
    val_slices = []

    for i, (sl, dim) in enumerate(zip(slices, dims)):
        # If val was not copied over that dim,
        # we need to take the appropriate subtensor on it.
        if i >= n_added_dims:
            # We check that the corresponding val dimensions was
            # not a broadcasted dimensions.
            if (
                val.type.ndim > (i - n_added_dims)
                and val.type.broadcastable[i - n_added_dims]
            ):
                val_slices.append(slice(None))
            else:
                val_slices.append(sl)

        csl, _ = get_canonical_form_slice(sl, dim)
        if type(csl) is not slice:
            # That dimension is removed.
            pass
        else:
            nw_dim = csl.stop - csl.start

            if csl.step != 1:
                # Do not add the ceil_intdiv() graphs in the graphs
                # when this is not needed as it prevent detecting the
                # correct broadcast pattern.
                nw_dim = ceil_intdiv(nw_dim, csl.step)
            nw_dims += [nw_dim]

    nw_val = val[tuple(val_slices)]
    nw_dims += dims[len(slices) :]
    if nw_val.ndim > len(nw_dims):
        return False
    rval = alloc(nw_val, *nw_dims)
    if not isinstance(rval, (list, tuple)):
        rval = [rval]
    return rval


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_inc_subtensor(fgraph, node):
    """
    Subtensor(SetSubtensor(x, y, idx), idx) -> y

    """
    if isinstance(node.op, Subtensor):
        x = node.inputs[0]
        if not x.owner or not isinstance(x.owner.op, IncSubtensor):
            return
        if not x.owner.op.set_instead_of_inc:
            return

        if x.owner.inputs[2:] == node.inputs[1:] and tuple(
            x.owner.op.idx_list
        ) == tuple(node.op.idx_list):
            out = node.outputs[0]
            y = x.owner.inputs[1]
            # If the dtypes differ, cast y into x.dtype
            if x.dtype != y.dtype:
                y = y.astype(x.dtype)
            if (
                out.type.dtype == y.type.dtype
                and out.type.broadcastable == y.type.broadcastable
            ):
                # if x[idx] and y have the same type, directly return y
                return [y]
            else:
                # The difference is related to broadcasting pattern
                assert out.broadcastable != y.broadcastable
                # We have to alloc y to the shape of x[idx]
                x_subtensor = node.op(x.owner.inputs[0], *x.owner.inputs[2:])
                return [alloc(y, *x_subtensor.shape)]
        else:
            return


@register_specialize
@register_canonicalize("fast_compile")
@register_useless
@node_rewriter([Subtensor, AdvancedSubtensor1])
def local_subtensor_make_vector(fgraph, node):
    """Perform ``*Subtensor*`` operations on ``MakeVector`` outputs when the indices are constant.

    Replace all ``Subtensor`` and ``MakeVector`` cases like:
        [a,b,c][0] -> a
        [a,b,c][0:2] -> [a,b]

    Replace all ``AdvancedSubtensor1`` and ``MakeVector`` cases like:
        [a,b,c][[0,2]] -> [a,c]

    We can do this for constant indexes.

    .. note:

        This optimization implicitly relies on shape optimizations.

    TODO: This only applies to a single indexed dimension; we should have
    something more general for constant ``*Subtensor*`` graphs (or perhaps
    include this kind of work in the constant folding).
    """

    if not isinstance(node.op, (Subtensor, AdvancedSubtensor1)):
        return False

    x = node.inputs[0]

    if not x.owner or not isinstance(x.owner.op, MakeVector):
        return False

    make_vector_op = x.owner.op

    if isinstance(node.op, Subtensor):
        (idx,) = node.op.idx_list

        if isinstance(idx, (aes.ScalarType, TensorType)):
            old_idx, idx = idx, node.inputs[1]
            assert idx.type.is_super(old_idx)
    elif isinstance(node.op, AdvancedSubtensor1):
        idx = node.inputs[1]

    if isinstance(idx, (int, np.integer)):
        return [x.owner.inputs[idx]]
    elif isinstance(idx, Variable):
        if idx.ndim == 0:
            try:
                v = get_scalar_constant_value(idx, only_process_constants=True)
                try:
                    ret = [x.owner.inputs[v]]
                except IndexError:
                    raise NotScalarConstantError("Bad user graph!")
                return ret
            except NotScalarConstantError:
                pass
        elif idx.ndim == 1 and isinstance(idx, Constant):
            values = list(map(int, list(idx.value)))
            ret = make_vector_op(*[x.owner.inputs[v] for v in values])
            copy_stack_trace(node.outputs[0], ret)
            return [ret]
    elif isinstance(idx, slice):
        # The index is a slice.  If it's a constant slice, we can perform the
        # index operation here.
        try:
            const_slice = get_constant_idx(
                node.op.idx_list, node.inputs, allow_partial=False
            )[0]
            ret = make_vector_op(*x.owner.inputs[const_slice])
            copy_stack_trace(node.outputs, ret)
            return [ret]
        except NotScalarConstantError:
            pass


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor])
def local_useless_inc_subtensor(fgraph, node):
    r"""Remove redundant `IncSubtensor`\s.

    More specifically, ``set_subtensor(x[indices], y)`` is replaced by
    ``y[indices]`` when ``indices`` are full `slice`\s and ``y``'s shape is
    equal to ``x[indices]``, and ``inc_subtensor(x[indices], y)`` is replaced
    by ``y[indices]`` when ``x[indices]`` is some array of ``0``\s, ``indices``
    are full slices, and the shapes are equal.
    """
    if not isinstance(node.op, IncSubtensor):
        return

    if not hasattr(fgraph, "shape_feature"):
        return

    x, y, *index_inputs = node.inputs

    if node.op.set_instead_of_inc is False:
        # This is an increment operation, so the array being incremented must
        # consist of all zeros in order for the entire operation to be useless
        try:
            c = get_scalar_constant_value(x)
            if c != 0:
                return
        except NotScalarConstantError:
            return

    idx_cst = indices_from_subtensor(list(index_inputs), node.op.idx_list)

    # Check that all indices are full slices with only reversals and no step
    # sizes
    # TODO: It seems like there should be a basic `IncSubtensor`
    # canonicalization that removes these redundant slices.
    if all(
        isinstance(e, slice)
        and e.start is None
        and e.stop is None
        and (
            e.step is None
            or extract_constant(e.step, only_process_constants=True) == -1
        )
        for e in idx_cst
    ):

        # `IncSubtensor` broadcasts `x` on `y` based on run-time shapes, so we
        # must check that they are the same
        if not fgraph.shape_feature.same_shape(x, y):
            return

        # There are no reversals, so we don't need a replacement.
        if all(e.step is None for e in node.op.idx_list):
            # They are exactly the same shapes, so we can remove this `IncSubtensor`
            return [y]

        new_node = Subtensor(node.op.idx_list).make_node(y, *index_inputs)
        new_out = new_node.outputs[0]
        copy_stack_trace(node.outputs, new_out)

        return [new_out]


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedIncSubtensor1])
def local_set_to_inc_subtensor(fgraph, node):
    r"""
    AdvancedIncSubtensor1(x, x[ilist]+other, ilist, set_instead_of_inc=True) ->
    AdvancedIncSubtensor1(x, other, ilist, set_instead_of_inc=False)

    TODO FIXME: Why doesn't this apply to all `*IncSubtensor*` `Op`\s?  If it
    did this wouldn't need to also be included in the "specialize" pass.

    """
    if (
        isinstance(node.op, AdvancedIncSubtensor1)
        and node.op.set_instead_of_inc
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Elemwise)
        and isinstance(node.inputs[1].owner.op.scalar_op, aes.Add)
    ):
        addn = node.inputs[1].owner
        subn = None
        other = None

        if addn.inputs[0].owner and isinstance(
            addn.inputs[0].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[0].owner
            other = addn.inputs[1]
        elif addn.inputs[1].owner and isinstance(
            addn.inputs[1].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[1].owner
            other = addn.inputs[0]
        else:
            return
        if subn.inputs[1] != node.inputs[2] or subn.inputs[0] != node.inputs[0]:
            return
        ret = advanced_inc_subtensor1(node.inputs[0], other, node.inputs[2])

        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_useless_subtensor(fgraph, node):
    """Remove `Subtensor` if it takes the full input."""
    # This optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_of = fgraph.shape_feature.shape_of

    cdata = get_constant_idx(
        node.op.idx_list,
        node.inputs,
        allow_partial=True,
        only_process_constants=True,
    )
    for pos, idx in enumerate(cdata):
        if not isinstance(idx, slice):
            # If idx is not a slice, this means we remove this dimension
            # from the output, so the subtensor is not useless
            return False
        if idx.start is not None and idx.start != 0:
            # If the start of the slice is different from 0, or is a
            # variable, then we assume the subtensor is not useless
            return False
        if idx.step is not None and idx.step != 1:
            # If we are going backwards, or skipping elements, then this
            # is not a useless subtensor
            return False

        length_pos = shape_of[node.inputs[0]][pos]

        if isinstance(idx.stop, (int, np.integer)):
            length_pos_data = sys.maxsize
            try:
                length_pos_data = get_scalar_constant_value(
                    length_pos, only_process_constants=True
                )
            except NotScalarConstantError:
                pass

            if idx.stop < length_pos_data:
                return False
        elif isinstance(idx.stop, Variable):
            length_pos_shape_i = idx.stop
            # length_pos is a tensor variable, but length_pos_shape_i
            # is a scalar variable. We try to see if they represent
            # the same underlying variable.
            if length_pos_shape_i.owner and isinstance(
                length_pos_shape_i.owner.op, ScalarFromTensor
            ):
                length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
            elif length_pos.owner and isinstance(length_pos.owner.op, TensorFromScalar):
                length_pos = length_pos.owner.inputs[0]
            else:
                # We did not find underlying variables of the same type
                return False

            # The type can be different: int32 vs int64. length_pos
            # should always be int64 as that is what the shape
            # tracker keep. Subtensor accept any scalar int{8,16,32,64}
            # as index type.
            assert str(length_pos.type.dtype) == "int64"
            assert str(length_pos_shape_i.type.dtype) in [
                "int8",
                "int16",
                "int32",
                "int64",
            ]

            # length_pos_shape_i cannot be None
            if length_pos_shape_i != length_pos:
                return False
        elif idx.stop is None:
            continue
        else:
            return False

    return [node.inputs[0]]


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedSubtensor1])
def local_useless_AdvancedSubtensor1(fgraph, node):
    """Remove `AdvancedSubtensor1` if it takes the full input.

    In the `AdvancedSubtensor1` case, the full input is taken when the indices
    are equivalent to ``arange(0, input.shape[0], 1)`` using either an explicit
    list/vector or the `ARange` `Op`.

    """
    # This optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_of = fgraph.shape_feature.shape_of

    # get length of the indexed tensor along the first axis
    try:
        length = get_scalar_constant_value(
            shape_of[node.inputs[0]][0], only_process_constants=True
        )
    except NotScalarConstantError:
        return False

    # get index (which must be a vector by definition)
    idx = node.inputs[1]

    # `idx` must be equivalent to [0,1,...,shape[0] - 1] to qualify for
    # this optimization
    if isinstance(idx, Constant):
        idx = idx.value
        if len(idx) != length:
            return False
        if np.any(idx != np.arange(length)):
            return False
    else:
        return False

    return [node.inputs[0]]


def merge_two_slices(fgraph, slice1, len1, slice2, len2):
    """
     This function merges two slices into a single slice. The code works on
     the assumption that:

     a) slice1 is actually a slice and not an index, while slice2
        can be just an index.

     b) the two slices **have been applied consecutively** on the same
        tensor

    The output slice is **not** in canonical form, but actually just a slice
    that can be applied to a tensor to produce the same output as applying
    the two consecutive slices.
    ``len1`` is the length of the tensor **before** applying the first slice,
    while ``len2`` is the length **after** applying the first slice.
    """

    if not isinstance(slice1, slice):
        raise ValueError("slice1 should be of type `slice`")

    sl1, reverse1 = get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = get_canonical_form_slice(slice2, len2)

    if not isinstance(sl2, slice):
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = switch(lt(reverse1, 0), n_val, p_val)
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = switch(
            and_(lt(pn_stop, 0), gt(flen, 0)),
            -len1 - 1,
            minimum(pn_stop, sl1.stop),
        )
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = minimum(pn_start, sl1.stop)
        pn_start = maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = switch(
            and_(lt(np_stop, 0), gt(flen, 0)),
            -len1 - 1,
            maximum(sl1.start - 1, np_stop),
        )
        np_start = maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = maximum(sl1.start, (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_start, pn_start),
            switch(lt(reverse1, 0), nn_start, pp_start),
        )

        stop = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_stop, pn_stop),
            switch(lt(reverse1, 0), nn_stop, pp_stop),
        )

        step = switch(lt(reverse2 * reverse1, 0), n_step, p_step)
        start = switch(le(flen, 0), 0, start)
        stop = switch(le(flen, 0), 0, stop)

        return slice(start, stop, step)


@register_canonicalize
@node_rewriter([add])
def local_IncSubtensor_serialize(fgraph, node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """

    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return (
            i.owner
            and isinstance(
                i.owner.op,
                (
                    IncSubtensor,
                    AdvancedIncSubtensor1,
                    AdvancedIncSubtensor,
                ),
            )
            and i.type.is_super(o_type)
            and len(fgraph.clients[i]) == 1
            and not i.owner.op.set_instead_of_inc
        )

    if node.op == add:
        o_type = node.outputs[0].type

        movable_inputs = [i for i in node.inputs if movable(i)]

        if movable_inputs:
            new_inputs = [i for i in node.inputs if not movable(i)] + [
                mi.owner.inputs[0] for mi in movable_inputs
            ]
            if len(new_inputs) == 0:
                new_add = new_inputs[0]
            else:
                new_add = add(*new_inputs)

                # Copy over stacktrace from original output, as an error
                # (e.g. an index error) in this add operation should
                # correspond to an error in the original add operation.
                copy_stack_trace(node.outputs[0], new_add)

            # stack up the new incsubtensors
            tip = new_add
            for mi in movable_inputs:
                assert o_type.is_super(tip.type)
                assert mi.owner.inputs[0].type.is_super(tip.type)
                tip = mi.owner.op(tip, *mi.owner.inputs[1:])
                # Copy over stacktrace from outputs of the original
                # "movable" operation to the new operation.
                copy_stack_trace(node.outputs + mi.owner.outputs, tip)

            return [tip]

        # print incsub_inputs, [id(i.owner.inputs[0]) for i in incsub_inputs]


# We register it in a WalkingGraphRewriter inside the canonizer EQ optimizer.
# Otherwise in some cases it was making the EQ optimizer use 45. In
# the WalkingGraphRewriter, the EQ only use 5 passes.
compile.optdb.register(
    "pre_local_IncSubtensor_serialize",
    in2out(local_IncSubtensor_serialize),
    "fast_run",
    # Just before canonizer
    position=0.99,
)


# after priority 50 Destructive inplace operations
# gemm is the first one now, at priority 70


@node_rewriter([IncSubtensor], inplace=True)
def local_inplace_setsubtensor(fgraph, node):
    if isinstance(node.op, IncSubtensor) and not node.op.inplace:
        dta = node.op.destroyhandler_tolerate_aliased
        new_op = node.op.__class__(
            node.op.idx_list,
            inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            destroyhandler_tolerate_aliased=dta,
        )
        new_node = new_op(*node.inputs)
        val = getattr(node.outputs[0].tag, "nan_guard_mode_check", True)
        new_node.tag.nan_guard_mode_check = val

        # Copy stacktrace from original outputs to new outputs.
        # This is sensible, because the new operation is the
        # same as the old one, but now with different attributes.
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_setsubtensor",
    WalkingGraphRewriter(
        local_inplace_setsubtensor, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=60,
)


@node_rewriter([AdvancedIncSubtensor1], inplace=True)
def local_inplace_AdvancedIncSubtensor1(fgraph, node):
    if isinstance(node.op, AdvancedIncSubtensor1) and not node.op.inplace:
        new_op = node.op.clone_inplace()
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor1",
    WalkingGraphRewriter(
        local_inplace_AdvancedIncSubtensor1,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=60,
)


@node_rewriter([AdvancedIncSubtensor], inplace=True)
def local_inplace_AdvancedIncSubtensor(fgraph, node):
    if isinstance(node.op, AdvancedIncSubtensor) and not node.op.inplace:
        new_op = type(node.op)(
            inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            ignore_duplicates=node.op.ignore_duplicates,
        )
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor",
    WalkingGraphRewriter(
        local_inplace_AdvancedIncSubtensor,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=60,
)


# Register old name
@register_canonicalize("local_incsubtensor_of_allocs")
@register_stabilize("local_incsubtensor_of_allocs")
@node_rewriter([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_incsubtensor_of_zeros(fgraph, node):
    """
    IncSubtensor(x, zeros, idx) -> x

    """
    if (
        isinstance(node.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))
        and not node.op.set_instead_of_inc
    ):
        x = node.inputs[0]
        y = node.inputs[1]
        try:
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            if get_scalar_constant_value(y, elemwise=False) == 0:
                # No need to copy over the stacktrace,
                # because x should already have a stacktrace
                return [x]
        except NotScalarConstantError:
            return


@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor])
def local_incsubtensor_of_zeros_to_setsubtensor(fgraph, node):
    """
    IncSubtensor(zeros, x, ...) -> SetSubtensor(zeros, x, ...)
    """
    if isinstance(node.op, (IncSubtensor)) and not node.op.set_instead_of_inc:
        x = node.inputs[0]

        if isinstance(x, Constant) and not np.any(x.data):
            return [
                IncSubtensor(
                    node.op.idx_list,
                    node.op.inplace,
                    set_instead_of_inc=True,
                    destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased,
                )(*node.inputs)
            ]


@register_canonicalize("local_setsubtensor_of_allocs")
@register_stabilize("local_setsubtensor_of_allocs")
@node_rewriter([IncSubtensor])
def local_setsubtensor_of_constants(fgraph, node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.

    """
    if isinstance(node.op, IncSubtensor) and node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]

        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        try:
            replace_x = get_scalar_constant_value(x, elemwise=False)
        except NotScalarConstantError:
            return

        try:
            replace_y = get_scalar_constant_value(y, elemwise=False)
        except NotScalarConstantError:
            return

        if replace_x == replace_y:

            # No need to copy over the stacktrace,
            # because x should already have a stacktrace
            return [x]
        else:
            return False


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedSubtensor1])
def local_adv_sub1_adv_inc_sub1(fgraph, node):
    """Rewrite graphs like ``AdvancedSubtensor1(AdvancedSetSubtensor1(...), ...)``.

    .. code::

        AdvancedSubtensor1(AdvancedSetSubtensor1(x, y, idx), idx) -> y


    Notes
    -----
    This rewrite adds an `AssertOp`; otherwise, it would remove shape and index
    error. If you want to get rid of them, see the :ref:`unsafe_rewrites`
    section.

    A previous version of this rewrite also matched
    ``AdvancedSubtensor1(AdvancedIncSubtensor1(x, y, idx), idx)``.
    This is incorrect when there are duplicate indices.
    The current version warns the user about potential issues.

    """
    if not isinstance(node.op, AdvancedSubtensor1):
        return
    inp = node.inputs[0]
    if not inp.owner or not isinstance(inp.owner.op, AdvancedIncSubtensor1):
        return
    idx = node.inputs[1]
    idx2 = inp.owner.inputs[2]
    x = inp.owner.inputs[0]
    y = inp.owner.inputs[1]
    if idx is not idx2:
        return
    if (
        not inp.owner.op.set_instead_of_inc
        and
        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        extract_constant(x, elemwise=False) != 0
    ):
        return

    if not inp.owner.op.set_instead_of_inc:
        return

    cond = [at_all(and_(lt(idx, x.shape[0]), ge(idx, -x.shape[0])))]
    if not fgraph.shape_feature.same_shape(idx, y, 0, 0):
        cond.append(eq(idx.shape[0], y.shape[0]))
    r = Assert(
        "Bad indexing or shapes in a AdvancedIncSubtensor1 " "that was optimized away"
    )(y, *cond)
    copy_stack_trace(y, r)

    if r.dtype == node.outputs[0].dtype:
        return [r]
    # It is possible that y is upcast or downcast to x.dtype.
    # In all case, as we set or add with 0, we can just cast y.
    r2 = cast(r, node.outputs[0].dtype)

    # Copy over stacktrace from before casting, since
    # we don't expect problems in the casting operation,
    # and any problems in the indexing would have been spotted above.
    copy_stack_trace(r, r2)
    return [r2]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@node_rewriter([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_useless_inc_subtensor_alloc(fgraph, node):
    """
    Replaces an [Advanced]IncSubtensor[1], whose increment is an `alloc` of
    a fully or partially broadcastable variable, by one that skips the
    intermediate `alloc` where possible.

    """
    if isinstance(node.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)):
        x = node.inputs[0]
        y = node.inputs[1]
        i = node.inputs[2:]

        if y.owner is not None and isinstance(y.owner.op, Alloc):
            # `z` is the input of the Alloc op, i.e. at.alloc(z, <shape>)
            z = y.owner.inputs[0]

            try:
                shape_feature = fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            shape_of = shape_feature.shape_of
            same_shape = shape_feature.same_shape

            # Get the subtensor of `x` indexed by `i` in order to compare
            # shapes later.
            if isinstance(node.op, IncSubtensor):
                xi = Subtensor(node.op.idx_list)(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = advanced_subtensor(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor1):
                xi = advanced_subtensor1(x, *i)
            else:
                raise Exception("Should never happen!")

            reason = "local_useless_incsubtensor_alloc"

            # Add `xi` to the shape feature `fgraph`. This is important for
            # shape inference later because the variable must be part of the
            # function graph in order to call `same_shape` on it.
            if xi not in shape_of:
                shape_feature.on_import(fgraph, xi.owner, f"{reason}: add `xi`")

            # `xi` may have more dimensions than `y` since the subtensor ops
            # do automatic broadcasting of the increment internally. Thus, we
            # need to make the leading implicitly broadcasted dimensions
            # explicit for shape comparison later.
            if xi.ndim > y.ndim:
                y = shape_padleft(y, xi.ndim - y.ndim)
                if y not in shape_of:
                    shape_feature.on_import(fgraph, y.owner, f"{reason}: add `y`")

            # Build `z_broad` explicitly to include extra implicit dimensions.
            z_broad = (True,) * (xi.ndim - z.ndim) + z.broadcastable

            cond = [
                # The shapes of `y` and `xi` must either agree or `y` may
                # also have shape equal to 1 which may be treated as a
                # broadcastable dimension by the subtensor op.
                or_(eq(y.shape[k], 1), eq(y.shape[k], xi.shape[k]))
                # Loop over all dimensions.
                for k in range(xi.ndim)
                # We need to check the above shapes, if
                # * the pre-alloc increment `z` is broadcastable in
                # dimension `k` (if it isn't, then the shapes of `z` and
                # `y` are the same by the definition of the `Alloc` op in
                # this dimension and replacing `y` by `z` will not hide a
                # shape error), and
                # * `xi` and `y` do not have the same shape in dimension
                # `k` or we cannot infer the shape statically (if the
                # shapes of `xi` and `y` are not the same, then replacing
                # `y` by `z` will hide the shape error of `y`), and
                # * the shape of `y` is not equal to 1 or we cannot infer
                # the shape statically (if the shape of `y` is equal to
                # 1, then `y` is broadcasted by the inc_subtensor op
                # internally, so the shapes of `xi` and `y` do not need
                # to match in dimension `k`; else we need to check at
                # runtime that the shape of `y` is either 1 or the same
                # as `xi` or otherwise replacing `y` by `z` will hide a
                # shape error).
                if (
                    z_broad[k]
                    and not same_shape(xi, y, dim_x=k, dim_y=k)
                    and shape_of[y][k] != 1
                )
            ]

            if len(cond) > 0:
                msg = "`x[i]` and `y` do not have the same shape."
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *i)
            # Copy over stacktrace from previous output, since
            # we don't expect problems when removing the intermediate
            # alloc operation and so we still want to point at the line
            # of the inc_subtensor operation.
            copy_stack_trace(node.outputs, r)

            return [r]


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_shape_constant(fgraph, node):
    r"""Simplify constant `Subtensor`\s on `Shape`\s dimensions that are known.

    We want to convert graphs like

        Subtensor{int64} [id A] ''
         |Shape [id B] ''
         | |<TensorType(float64, row)> [id C]
         |ScalarConstant{0} [id D]

    into

        TensorConstant{1}

    TODO: Something like `local_shape_to_shape_i` should be a general
    canonicalization, and not a `ShapeFeature`-dependent rewrite.  If that were
    the case, we could change this to only operate on `Shape_i`\s.
    Currently, we're not handling them because they should only appear when
    `ShapeFeature` is present, and it will also simplify/remove them.

    """
    if not isinstance(node.op, Subtensor):
        return False

    shape = node.inputs[0]

    if not (shape.owner and isinstance(shape.owner.op, Shape)):
        return False

    shape_arg = shape.owner.inputs[0]

    (idx,) = get_idx_list(node.inputs, node.op.idx_list)

    try:
        idx_val = as_index_literal(idx)
    except NotScalarConstantError:
        return False

    assert idx_val != np.newaxis

    if not isinstance(shape_arg.type, TensorType):
        return False

    shape_parts = shape_arg.type.broadcastable[idx_val]

    if isinstance(shape_parts, Iterable):
        if all(shape_parts):
            return [as_tensor([1] * len(shape_parts), dtype=np.int64, ndim=1)]
    elif shape_parts:
        return [as_tensor(1, dtype=np.int64)]


@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_SpecifyShape_lift(fgraph, node):
    """Lift ``specify_shape(x, s)[i_1, ..., i_n]`` to ``specify_shape(x[i1, ... , i_n], s[n:])``."""

    if not isinstance(node.op, Subtensor):
        return False

    specify_shape_node = node.inputs[0]

    if not (
        specify_shape_node.owner
        and isinstance(specify_shape_node.owner.op, SpecifyShape)
    ):
        return False

    obj_arg = specify_shape_node.owner.inputs[0]
    shape_arg = specify_shape_node.owner.inputs[1:]

    indices = get_idx_list(node.inputs, node.op.idx_list)

    if any(
        isinstance(index, slice) or isinstance(getattr(index, "type", None), SliceType)
        for index in indices
    ):
        return False

    new_obj_arg = obj_arg[indices]
    # No need to specify shape for scalar outputs
    if new_obj_arg.ndim == 0:
        return [new_obj_arg]
    return [specify_shape(new_obj_arg, shape_arg[len(indices) :])]


@register_specialize
@node_rewriter([Join])
def local_join_subtensors(fgraph, node):
    r"""Simplify contiguous :class:`Subtensor`\s inside a :class:`Join`.

    `join((x[:3], x[3:5]), axis=0) -> x[:5]`
    """
    # TODO: Generalize to AdvancedSubtensors

    axis, tensors = node.inputs[0], node.inputs[1:]

    try:
        axis = get_scalar_constant_value(axis)
    except NotScalarConstantError:
        return

    for subtensor1_idx, (subtensor1, subtensor2) in enumerate(
        zip(tensors[:-1], tensors[1:])
    ):
        # Check that two consecutive Subtensors are operating on the same base tensor
        if not (
            (
                subtensor1.owner is not None
                and isinstance(subtensor1.owner.op, Subtensor)
            )
            and (
                subtensor2.owner is not None
                and isinstance(subtensor2.owner.op, Subtensor)
            )
            and (subtensor1.owner.inputs[0] is subtensor2.owner.inputs[0])
        ):
            continue

        # Check that subtensors have consecutive indexes across the join axis
        idxs_subtensor1 = indices_from_subtensor(
            subtensor1.owner.inputs[1:], subtensor1.owner.op.idx_list
        )
        idxs_subtensor2 = indices_from_subtensor(
            subtensor2.owner.inputs[1:], subtensor2.owner.op.idx_list
        )
        try:
            idxs_axis_subtensor1 = idxs_subtensor1[axis]
            idxs_axis_subtensor2 = idxs_subtensor2[axis]
        except IndexError:
            continue
        if not (
            isinstance(idxs_axis_subtensor1, slice)
            and isinstance(idxs_axis_subtensor2, slice)
        ):
            continue
        start_subtensor1, stop_subtensor1, step_subtensor1 = (
            idxs_axis_subtensor1.start,
            idxs_axis_subtensor1.stop,
            idxs_axis_subtensor1.step,
        )
        start_subtensor2, stop_subtensor2, step_subtensor2 = (
            idxs_axis_subtensor2.start,
            idxs_axis_subtensor2.stop,
            idxs_axis_subtensor2.step,
        )
        if not (
            (stop_subtensor1 is not None and start_subtensor2 is not None)
            and (stop_subtensor1 == start_subtensor2)
        ):
            continue

        # Check that step is None or 1
        # For non-unit steps (perhaps except for -1) we would need to know the
        # exact values of start and stop to know if they can be merged
        for step in (step_subtensor1, step_subtensor2):
            if step is None:
                continue
            try:
                if get_scalar_constant_value(step, only_process_constants=True) != 1:
                    return None
            except NotScalarConstantError:
                return None

        # Check that all other idxs of subtensor are the same
        if all(
            idxs_nonaxis_subtensor1 == idxs_nonaxis_subtensor2
            for i, (idxs_nonaxis_subtensor1, idxs_nonaxis_subtensor2) in enumerate(
                zip(idxs_subtensor1, idxs_subtensor2)
            )
            if i != axis
        ):

            base_tensor = subtensor1.owner.inputs[0]
            new_idxs = list(idxs_subtensor1)
            new_idxs[axis] = slice(start_subtensor1, stop_subtensor2, step_subtensor1)
            merged_subtensors = base_tensor[new_idxs]

            new_joined_tensors = [
                *tensors[:subtensor1_idx],
                merged_subtensors,
                *tensors[subtensor1_idx + 2 :],
            ]
            if len(new_joined_tensors) > 1:
                return [concatenate(new_joined_tensors, axis=axis)]
            else:
                return [merged_subtensors]


@register_specialize
@node_rewriter(
    [
        Subtensor,
        AdvancedSubtensor1,
        AdvancedSubtensor,
        IncSubtensor,
        AdvancedIncSubtensor,
        AdvancedIncSubtensor1,
    ]
)
def local_uint_constant_indices(fgraph, node):
    """Convert constant indices to unsigned dtypes."""

    if isinstance(node.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)):
        x, y, *indices = node.inputs
    else:
        x, *indices = node.inputs
        y = None

    idx_list = getattr(node.op, "idx_list", None)
    new_indices = list(indices_from_subtensor(indices, idx_list))
    has_new_index = False

    for i, index in enumerate(new_indices):

        if not isinstance(index, Constant):
            continue

        index_val = index.data

        if index_val is None or isinstance(index_val, slice):
            # TODO: If slice index dtypes matter, we can consider converting
            # those, as well.
            continue

        assert isinstance(index_val, (np.generic, np.ndarray))

        if index_val.size == 0:
            continue

        if index_val.dtype == bool:
            continue

        if np.ndim(index_val) > 0:
            minval = index_val.min()
        else:
            minval = index_val

        if minval >= 0:
            maxval = index_val.max()
            dtype = np.min_scalar_type(maxval)
        else:
            # If we can't convert to unsigned, then don't attempt to minimize
            # the type size either--at least not for now.
            # dtype = np.min_scalar_type(-max(-minval, maxval))
            continue

        if dtype == index_val.dtype:
            continue

        if index_val.ndim > 0:
            new_index = aesara.tensor.as_tensor_variable(
                index_val.astype(dtype), dtype=dtype
            )
        else:
            new_index = aes.constant(index_val.astype(dtype), dtype=dtype)

        new_indices[i] = new_index
        has_new_index = True

    if not has_new_index:
        return False

    new_out = x[tuple(new_indices)]

    if y is not None:
        new_out = inc_subtensor(
            new_out,
            y,
            inplace=node.op.inplace,
            set_instead_of_inc=node.op.set_instead_of_inc,
            ignore_duplicates=getattr(node.op, "ignore_duplicates", False),
        )

    new_outs = new_out.owner.outputs
    copy_stack_trace(node.outputs, new_outs)

    return new_outs

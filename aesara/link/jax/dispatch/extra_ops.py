import warnings

import jax.numpy as jnp

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.extra_ops import (
    Bartlett,
    BroadcastTo,
    CumOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    Repeat,
    Unique,
    UnravelIndex,
)


@jax_funcify.register(Bartlett)
def jax_funcify_Bartlett(op, **kwargs):
    def bartlett(x):
        return jnp.bartlett(x)

    return bartlett


@jax_funcify.register(CumOp)
def jax_funcify_CumOp(op, **kwargs):
    axis = op.axis
    mode = op.mode

    def cumop(x, axis=axis, mode=mode):
        if mode == "add":
            return jnp.cumsum(x, axis=axis)
        else:
            return jnp.cumprod(x, axis=axis)

    return cumop


@jax_funcify.register(Repeat)
def jax_funcify_Repeat(op, **kwargs):
    axis = op.axis

    def repeatop(x, repeats, axis=axis):
        return jnp.repeat(x, repeats, axis=axis)

    return repeatop


@jax_funcify.register(Unique)
def jax_funcify_Unique(op, **kwargs):
    axis = op.axis

    if axis is not None:
        raise NotImplementedError(
            "jax.numpy.unique is not implemented for the axis argument"
        )

    return_index = op.return_index
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    def unique(
        x,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    ):
        ret = jnp.lax_numpy._unique1d(x, return_index, return_inverse, return_counts)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    return unique


@jax_funcify.register(UnravelIndex)
def jax_funcify_UnravelIndex(op, **kwargs):
    order = op.order

    warnings.warn("JAX ignores the `order` parameter in `unravel_index`.")

    def unravelindex(indices, dims, order=order):
        return jnp.unravel_index(indices, dims)

    return unravelindex


@jax_funcify.register(RavelMultiIndex)
def jax_funcify_RavelMultiIndex(op, **kwargs):
    mode = op.mode
    order = op.order

    def ravelmultiindex(*inp, mode=mode, order=order):
        multi_index, dims = inp[:-1], inp[-1]
        return jnp.ravel_multi_index(multi_index, dims, mode=mode, order=order)

    return ravelmultiindex


@jax_funcify.register(BroadcastTo)
def jax_funcify_BroadcastTo(op, **kwargs):
    def broadcast_to(x, *shape):
        return jnp.broadcast_to(x, shape)

    return broadcast_to


@jax_funcify.register(FillDiagonal)
def jax_funcify_FillDiagonal(op, **kwargs):
    def filldiagonal(value, diagonal):
        i, j = jnp.diag_indices(min(value.shape[-2:]))
        return value.at[..., i, j].set(diagonal)

    return filldiagonal


@jax_funcify.register(FillDiagonalOffset)
def jax_funcify_FillDiagonalOffset(op, **kwargs):

    # def filldiagonaloffset(a, val, offset):
    #     height, width = a.shape
    #
    #     if offset >= 0:
    #         start = offset
    #         num_of_step = min(min(width, height), width - offset)
    #     else:
    #         start = -offset * a.shape[1]
    #         num_of_step = min(min(width, height), height + offset)
    #
    #     step = a.shape[1] + 1
    #     end = start + step * num_of_step
    #     a.flat[start:end:step] = val
    #
    #     return a
    #
    # return filldiagonaloffset

    raise NotImplementedError("flatiter not implemented in JAX")

import jax
import jax.numpy as jnp

from aesara.link.jax.dispatch.basic import jax_funcify, jnp_safe_copy
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@jax_funcify.register(Elemwise)
def jax_funcify_Elemwise(op, **kwargs):
    scalar_op = op.scalar_op
    return jax_funcify(scalar_op, **kwargs)


@jax_funcify.register(CAReduce)
def jax_funcify_CAReduce(op, **kwargs):
    axis = op.axis
    op_nfunc_spec = getattr(op, "nfunc_spec", None)
    scalar_nfunc_spec = getattr(op.scalar_op, "nfunc_spec", None)
    scalar_op_name = getattr(op.scalar_op, "name", None)
    scalar_op_identity = getattr(op.scalar_op, "identity", None)
    acc_dtype = getattr(op, "acc_dtype", None)

    def careduce(x):
        nonlocal axis, op_nfunc_spec, scalar_nfunc_spec, scalar_op_name, scalar_op_identity, acc_dtype

        if axis is None:
            axis = list(range(x.ndim))

        if acc_dtype is None:
            acc_dtype = x.dtype.type

        if op_nfunc_spec:
            jax_op = getattr(jnp, op_nfunc_spec[0])
            return jax_op(x, axis=axis).astype(acc_dtype)

        # The Aesara `Op` didn't tell us which NumPy equivalent to use (or
        # there isn't one), so we use this fallback approach
        if scalar_nfunc_spec:
            scalar_fn_name = scalar_nfunc_spec[0]
        elif scalar_op_name:
            scalar_fn_name = scalar_op_name

        to_reduce = reversed(sorted(axis))

        if to_reduce:
            # In this case, we need to use the `jax.lax` function (if there
            # is one), and not the `jnp` version.
            jax_op = getattr(jax.lax, scalar_fn_name)
            init_value = jnp.array(scalar_op_identity, dtype=acc_dtype)
            return jax.lax.reduce(x, init_value, jax_op, to_reduce).astype(acc_dtype)
        else:
            return x

    return careduce


@jax_funcify.register(DimShuffle)
def jax_funcify_DimShuffle(op, **kwargs):
    def dimshuffle(x):

        res = jnp.transpose(x, op.transposition)

        shape = list(res.shape[: len(op.shuffle)])

        for augm in op.augment:
            shape.insert(augm, 1)

        res = jnp.reshape(res, shape)

        if not op.inplace:
            res = jnp_safe_copy(res)

        return res

    return dimshuffle


@jax_funcify.register(Softmax)
def jax_funcify_Softmax(op, **kwargs):
    axis = op.axis

    def softmax(x):
        return jax.nn.softmax(x, axis=axis)

    return softmax


@jax_funcify.register(SoftmaxGrad)
def jax_funcify_SoftmaxGrad(op, **kwargs):
    axis = op.axis

    def softmax_grad(dy, sm):
        dy_times_sm = dy * sm
        return dy_times_sm - jnp.sum(dy_times_sm, axis=axis, keepdims=True) * sm

    return softmax_grad


@jax_funcify.register(LogSoftmax)
def jax_funcify_LogSoftmax(op, **kwargs):
    axis = op.axis

    def log_softmax(x):
        return jax.nn.log_softmax(x, axis=axis)

    return log_softmax

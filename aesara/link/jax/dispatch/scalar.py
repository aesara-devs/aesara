import functools

import jax
import jax.numpy as jnp

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.scalar import Softplus
from aesara.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Identity,
    IntDiv,
    Mod,
    Mul,
    ScalarOp,
    Second,
    Sub,
)
from aesara.scalar.math import Erf, Erfc, Erfinv, Log1mexp, Psi


def check_if_inputs_scalars(node):
    """Check whether all the inputs of an `Elemwise` are scalar values.

    `jax.lax` or `jax.numpy` functions systematically return `TracedArrays`,
    while the corresponding Python operators return concrete values when passed
    concrete values. In order to be able to compile the largest number of graphs
    possible we need to preserve concrete values whenever we can. We thus need
    to dispatch differently the Aesara operators depending on whether the inputs
    are scalars.

    """
    ndims_input = [inp.type.ndim for inp in node.inputs]
    are_inputs_scalars = True
    for ndim in ndims_input:
        try:
            if ndim > 0:
                are_inputs_scalars = False
        except TypeError:
            are_inputs_scalars = False

    return are_inputs_scalars


@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op, node, **kwargs):
    func_name = op.nfunc_spec[0]

    # We dispatch some Aesara operators to Python operators
    # whenever the inputs are all scalars.
    are_inputs_scalars = check_if_inputs_scalars(node)
    if are_inputs_scalars:
        elemwise = elemwise_scalar(op)
        if elemwise is not None:
            return elemwise

    if "." in func_name:
        jnp_func = functools.reduce(getattr, [jax] + func_name.split("."))
    else:
        jnp_func = getattr(jnp, func_name)

    if hasattr(op, "nfunc_variadic"):
        # These are special cases that handle invalid arities due to the broken
        # Aesara `Op` type contract (e.g. binary `Op`s that also function as
        # their own variadic counterparts--even when those counterparts already
        # exist as independent `Op`s).
        jax_variadic_func = getattr(jnp, op.nfunc_variadic)

        def elemwise(*args):
            if len(args) > op.nfunc_spec[1]:
                return jax_variadic_func(
                    jnp.stack(jnp.broadcast_arrays(*args), axis=0), axis=0
                )
            else:
                return jnp_func(*args)

        return elemwise
    else:
        return jnp_func


@functools.singledispatch
def elemwise_scalar(op):
    return None


@elemwise_scalar.register(Add)
def elemwise_scalar_add(op):
    def elemwise(*inputs):
        return sum(inputs)

    return elemwise


@elemwise_scalar.register(Mul)
def elemwise_scalar_mul(op):
    import operator
    from functools import reduce

    def elemwise(*inputs):
        return reduce(operator.mul, inputs, 1)

    return elemwise


@elemwise_scalar.register(Sub)
def elemwise_scalar_sub(op):
    def elemwise(x, y):
        return x - y

    return elemwise


@elemwise_scalar.register(IntDiv)
def elemwise_scalar_intdiv(op):
    def elemwise(x, y):
        return x // y

    return elemwise


@elemwise_scalar.register(Mod)
def elemwise_scalar_mod(op):
    def elemwise(x, y):
        return x % y

    return elemwise


@jax_funcify.register(Cast)
def jax_funcify_Cast(op, **kwargs):
    def cast(x):
        return jnp.array(x).astype(op.o_type.dtype)

    return cast


@jax_funcify.register(Identity)
def jax_funcify_Identity(op, **kwargs):
    def identity(x):
        return x

    return identity


@jax_funcify.register(Clip)
def jax_funcify_Clip(op, **kwargs):
    """Register the translation for the `Clip` `Op`.

    Aesara's `Clip` operator operates differently from NumPy's when the
    specified `min` is larger than the `max` so we cannot reuse `jax.numpy.clip`
    to maintain consistency with Aesara.

    """

    def clip(x, min, max):
        return jnp.where(x < min, min, jnp.where(x > max, max, x))

    return clip


@jax_funcify.register(Composite)
def jax_funcify_Composite(op, vectorize=True, **kwargs):
    jax_impl = jax_funcify(op.fgraph)

    def composite(*args):
        return jax_impl(*args)[0]

    return jnp.vectorize(composite)


@jax_funcify.register(Second)
def jax_funcify_Second(op, **kwargs):
    def second(x, y):
        return jnp.broadcast_to(y, x.shape)

    return second


@jax_funcify.register(Erf)
def jax_funcify_Erf(op, node, **kwargs):
    def erf(x):
        return jax.scipy.special.erf(x)

    return erf


@jax_funcify.register(Erfc)
def jax_funcify_Erfc(op, **kwargs):
    def erfc(x):
        return jax.scipy.special.erfc(x)

    return erfc


@jax_funcify.register(Erfinv)
def jax_funcify_Erfinv(op, **kwargs):
    def erfinv(x):
        return jax.scipy.special.erfinv(x)

    return erfinv


@jax_funcify.register(Log1mexp)
def jax_funcify_Log1mexp(op, node, **kwargs):
    def log1mexp(x):
        return jnp.where(
            x < jnp.log(0.5), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x))
        )

    return log1mexp


@jax_funcify.register(Psi)
def jax_funcify_Psi(op, node, **kwargs):
    def psi(x):
        return jax.scipy.special.digamma(x)

    return psi


@jax_funcify.register(Softplus)
def jax_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return jnp.where(
            x < -37.0,
            jnp.exp(x),
            jnp.where(
                x < 18.0,
                jnp.log1p(jnp.exp(x)),
                jnp.where(
                    x < 33.3,
                    x + jnp.exp(-x),
                    x,
                ),
            ),
        )

    return softplus

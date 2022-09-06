import functools

import jax
import jax.numpy as jnp

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.scalar import Softplus
from aesara.scalar.basic import Cast, Clip, Composite, Identity, ScalarOp, Second
from aesara.scalar.math import Erf, Erfc, Erfinv, Log1mexp, Psi


@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op, **kwargs):
    func_name = op.nfunc_spec[0]

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
        # This expression is numerically equivalent to the Aesara one
        # It just contains one "speed" optimization less than the Aesara counterpart
        return jnp.where(
            x < -37.0, jnp.exp(x), jnp.where(x > 33.3, x, jnp.log1p(jnp.exp(x)))
        )

    return softplus

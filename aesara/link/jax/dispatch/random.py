from functools import singledispatch

import jax
from numpy.random import Generator, RandomState
from numpy.random.bit_generator import (  # type: ignore[attr-defined]
    _coerce_to_uint32_array,
)

import aesara.tensor.random.basic as aer
from aesara.link.jax.dispatch.basic import jax_funcify, jax_typify
from aesara.link.jax.dispatch.shape import JAXShapeTuple
from aesara.tensor.shape import Shape, Shape_i


numpy_bit_gens = {"MT19937": 0, "PCG64": 1, "Philox": 2, "SFC64": 3}


SIZE_NOT_COMPATIBLE = """JAX random variables require concrete values for the `size` parameter of the distributions.
Concrete values are either constants:

>>> import aesara.tensor as at
>>> x_rv = at.random.normal(0, 1, size=(3, 2))

or the shape of an array:

>>> m = at.matrix()
>>> x_rv = at.random.normal(0, 1, size=m.shape)
"""


def assert_size_argument_jax_compatible(node):
    """Assert whether the current node can be JIT-compiled by JAX.

    JAX can JIT-compile `jax.random` functions when the `size` argument
    is a concrete value, i.e. either a constant or the shape of any
    traced value.

    """
    size = node.inputs[1]
    size_op = size.owner.op
    if not isinstance(size_op, (Shape, Shape_i, JAXShapeTuple)):
        raise NotImplementedError(SIZE_NOT_COMPATIBLE)


@jax_typify.register(RandomState)
def jax_typify_RandomState(state, **kwargs):
    state = state.get_state(legacy=False)
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]
    # XXX: Is this a reasonable approach?
    state["jax_state"] = state["state"]["key"][0:2]
    return state


@jax_typify.register(Generator)
def jax_typify_Generator(rng, **kwargs):
    state = rng.__getstate__()
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]

    # XXX: Is this a reasonable approach?
    state["jax_state"] = _coerce_to_uint32_array(state["state"]["state"])[0:2]

    # The "state" and "inc" values in a NumPy `Generator` are 128 bits, which
    # JAX can't handle, so we split these values into arrays of 32 bit integers
    # and then combine the first two into a single 64 bit integers.
    #
    # XXX: Depending on how we expect these values to be used, is this approach
    # reasonable?
    #
    # TODO: We might as well remove these altogether, since this conversion
    # should only occur once (e.g. when the graph is converted/JAX-compiled),
    # and, from then on, we use the custom "jax_state" value.
    inc_32 = _coerce_to_uint32_array(state["state"]["inc"])
    state_32 = _coerce_to_uint32_array(state["state"]["state"])
    state["state"]["inc"] = inc_32[0] << 32 | inc_32[1]
    state["state"]["state"] = state_32[0] << 32 | state_32[1]
    return state


@jax_funcify.register(aer.RandomVariable)
def jax_funcify_RandomVariable(op, node, **kwargs):
    """JAX implementation of random variables."""
    rv = node.outputs[1]
    out_dtype = rv.type.dtype
    out_size = rv.type.shape

    if isinstance(op, aer.MvNormalRV):
        # Aesara sets the `size` to the concatenation of the support shape
        # and the batch shape, while JAX explicitly requires the batch
        # shape only for the multivariate normal.
        out_size = node.outputs[1].type.shape[:-1]

    # If one dimension has unknown size, either the size is determined
    # by a `Shape` operator in which case JAX will compile, or it is
    # not and we fail gracefully.
    if None in out_size:
        assert_size_argument_jax_compatible(node)

        def sample_fn(rng, size, dtype, *parameters):
            return jax_sample_fn(op)(rng, size, out_dtype, *parameters)

    else:

        def sample_fn(rng, size, dtype, *parameters):
            return jax_sample_fn(op)(rng, out_size, out_dtype, *parameters)

    return sample_fn


@singledispatch
def jax_sample_fn(op):
    name = op.name
    raise NotImplementedError(
        f"No JAX implementation for the given distribution: {name}"
    )


@jax_sample_fn.register(aer.BetaRV)
@jax_sample_fn.register(aer.DirichletRV)
@jax_sample_fn.register(aer.PoissonRV)
@jax_sample_fn.register(aer.MvNormalRV)
def jax_sample_fn_generic(op):
    """Generic JAX implementation of random variables."""
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        sample = jax_op(sampling_key, *parameters, shape=size, dtype=dtype)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.CauchyRV)
@jax_sample_fn.register(aer.GumbelRV)
@jax_sample_fn.register(aer.LaplaceRV)
@jax_sample_fn.register(aer.LogisticRV)
@jax_sample_fn.register(aer.NormalRV)
@jax_sample_fn.register(aer.StandardNormalRV)
def jax_sample_fn_loc_scale(op):
    """JAX implementation of random variables in the loc-scale families.

    JAX only implements the standard version of random variables in the
    loc-scale family. We thus need to translate and rescale the results
    manually.

    """
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        loc, scale = parameters
        sample = loc + jax_op(sampling_key, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.BernoulliRV)
@jax_sample_fn.register(aer.CategoricalRV)
def jax_sample_fn_no_dtype(op):
    """Generic JAX implementation of random variables."""
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        sample = jax_op(sampling_key, *parameters, shape=size)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.RandIntRV)
@jax_sample_fn.register(aer.UniformRV)
def jax_sample_fn_uniform(op):
    """JAX implementation of random variables with uniform density.

    We need to pass the arguments as keyword arguments since the order
    of arguments is not the same.

    """
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        minval, maxval = parameters
        sample = jax_op(
            sampling_key, shape=size, dtype=dtype, minval=minval, maxval=maxval
        )
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.ParetoRV)
@jax_sample_fn.register(aer.GammaRV)
def jax_sample_fn_shape_rate(op):
    """JAX implementation of random variables in the shape-rate family.

    JAX only implements the standard version of random variables in the
    shape-rate family. We thus need to rescale the results manually.

    """
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (shape, rate) = parameters
        sample = jax_op(sampling_key, shape, size, dtype) / rate
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.ExponentialRV)
def jax_sample_fn_exponential(op):
    """JAX implementation of `ExponentialRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (scale,) = parameters
        sample = jax.random.exponential(sampling_key, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.StudentTRV)
def jax_sample_fn_t(op):
    """JAX implementation of `StudentTRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (
            df,
            loc,
            scale,
        ) = parameters
        sample = loc + jax.random.t(sampling_key, df, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.HalfNormalRV)
def jax_sample_fn_halfnormal(op):
    """JAX implementation of `HalfNormalRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (
            loc,
            scale,
        ) = parameters
        sample = (
            loc
            + jax.random.truncated_normal(sampling_key, 0.0, jax.numpy.inf, size, dtype)
            * scale
        )
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.HalfCauchyRV)
def jax_sample_fn_halfcauchy(op):
    """JAX implementation of `HalfCauchyRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (
            loc,
            scale,
        ) = parameters
        sample = loc + jax.numpy.abs(
            jax.random.cauchy(sampling_key, size, dtype) * scale
        )
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.ChoiceRV)
def jax_funcify_choice(op):
    """JAX implementation of `ChoiceRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (a, p, replace) = parameters
        smpl_value = jax.random.choice(sampling_key, a, size, replace, p)
        rng["jax_state"] = rng_key
        return (rng, smpl_value)

    return sample_fn


@jax_sample_fn.register(aer.PermutationRV)
def jax_sample_fn_permutation(op):
    """JAX implementation of `PermutationRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (x,) = parameters
        sample = jax.random.permutation(sampling_key, x)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(aer.LogNormalRV)
def jax_sample_fn_lognormal(op):
    """JAX implementation of `LogNormalRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        loc, scale = parameters
        sample = loc + jax.random.normal(sampling_key, size, dtype) * scale
        sample_exp = jax.numpy.exp(sample)
        rng["jax_state"] = rng_key
        return (rng, sample_exp)

    return sample_fn

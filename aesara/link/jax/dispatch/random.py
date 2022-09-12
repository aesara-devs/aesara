import jax
import jax.numpy as jnp
from numpy.random import Generator, RandomState
from numpy.random.bit_generator import (  # type: ignore[attr-defined]
    _coerce_to_uint32_array,
)

from aesara.link.jax.dispatch.basic import jax_funcify, jax_typify
from aesara.tensor.random.op import RandomVariable


numpy_bit_gens = {"MT19937": 0, "PCG64": 1, "Philox": 2, "SFC64": 3}


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


@jax_funcify.register(RandomVariable)
def jax_funcify_RandomVariable(op, node, **kwargs):
    name = op.name

    # TODO Make sure there's a 1-to-1 correspondance with names
    if not hasattr(jax.random, name):
        raise NotImplementedError(
            f"No JAX conversion for the given distribution: {name}"
        )

    dtype = node.outputs[1].dtype

    def random_variable(rng, size, dtype_num, *args):
        if not op.inplace:
            rng = rng.copy()
        prng = rng["jax_state"]
        data = getattr(jax.random, name)(key=prng, shape=size)
        smpl_value = jnp.array(data, dtype=dtype)
        rng["jax_state"] = jax.random.split(prng, num=1)[0]
        return (rng, smpl_value)

    return random_variable

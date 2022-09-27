from textwrap import dedent, indent
from typing import Any, Callable, Dict, Optional

import numba.np.unsafe.ndarray as numba_ndarray
import numpy as np
from numba import _helperlib, types
from numba.core import cgutils
from numba.extending import NativeValue, box, models, register_model, typeof_impl, unbox
from numpy.random import RandomState

import aesara.tensor.random.basic as aer
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import numba_funcify, numba_typify
from aesara.link.utils import (
    compile_function_src,
    get_name_for_object,
    unique_name_generator,
)
from aesara.tensor.basic import get_vector_length
from aesara.tensor.random.type import RandomStateType
from aesara.tensor.random.var import RandomStateSharedVariable


class RandomStateNumbaType(types.Type):
    def __init__(self):
        super(RandomStateNumbaType, self).__init__(name="RandomState")


random_state_numba_type = RandomStateNumbaType()


@typeof_impl.register(RandomState)
def typeof_index(val, c):
    return random_state_numba_type


@register_model(RandomStateNumbaType)
class RandomStateNumbaModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            # TODO: We can add support for boxing and unboxing
            # the attributes that describe a RandomState so that
            # they can be accessed inside njit functions, if required.
            ("state_key", types.Array(types.uint32, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@unbox(RandomStateNumbaType)
def unbox_random_state(typ, obj, c):
    """Convert a `RandomState` object to a native `RandomStateNumbaModel` structure.

    Note that this will create a 'fake' structure which will just get the
    `RandomState` objects accepted in Numba functions but the actual information
    of the Numba's random state is stored internally and can be accessed
    anytime using ``numba._helperlib.rnd_get_np_state_ptr()``.
    """
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)


@box(RandomStateNumbaType)
def box_random_state(typ, val, c):
    """Convert a native `RandomStateNumbaModel` structure to an `RandomState` object
    using Numba's internal state array.

    Note that `RandomStateNumbaModel` is just a placeholder structure with no
    inherent information about Numba internal random state, all that information
    is instead retrieved from Numba using ``_helperlib.rnd_get_state()`` and a new
    `RandomState` is constructed using the Numba's current internal state.
    """
    pos, state_list = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    rng = RandomState()
    rng.set_state(("MT19937", state_list, pos))
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(rng))
    return class_obj


@numba_typify.register(RandomState)
def numba_typify_RandomState(state, **kwargs):
    # The numba_typify in this case is just an passthrough function
    # that synchronizes Numba's internal random state with the current
    # RandomState object
    ints, index = state.get_state()[1:3]
    ptr = _helperlib.rnd_get_np_state_ptr()
    _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
    return state


def make_numba_random_fn(node, np_random_func):
    """Create Numba implementations for existing Numba-supported ``np.random`` functions.

    The functions generated here add parameter broadcasting and the ``size``
    argument to the Numba-supported scalar ``np.random`` functions.
    """
    if not isinstance(node.inputs[0], (RandomStateType, RandomStateSharedVariable)):
        raise TypeError("Numba does not support NumPy `Generator`s")

    tuple_size = int(get_vector_length(node.inputs[1]))
    size_dims = tuple_size - max(i.ndim for i in node.inputs[3:])

    # Make a broadcast-capable version of the Numba supported scalar sampling
    # function
    bcast_fn_name = f"aesara_random_{get_name_for_object(np_random_func)}"

    sized_fn_name = "sized_random_variable"

    unique_names = unique_name_generator(
        [
            bcast_fn_name,
            sized_fn_name,
            "np",
            "np_random_func",
            "numba_vectorize",
            "to_fixed_tuple",
            "tuple_size",
            "size_dims",
            "rng",
            "size",
            "dtype",
        ],
        suffix_sep="_",
    )

    bcast_fn_input_names = ", ".join(
        [unique_names(i, force_unique=True) for i in node.inputs[3:]]
    )
    bcast_fn_global_env = {
        "np_random_func": np_random_func,
        "numba_vectorize": numba_basic.numba_vectorize,
    }

    bcast_fn_src = f"""
@numba_vectorize
def {bcast_fn_name}({bcast_fn_input_names}):
    return np_random_func({bcast_fn_input_names})
    """
    bcast_fn = compile_function_src(
        bcast_fn_src, bcast_fn_name, {**globals(), **bcast_fn_global_env}
    )

    random_fn_input_names = ", ".join(
        ["rng", "size", "dtype"] + [unique_names(i) for i in node.inputs[3:]]
    )

    # Now, create a Numba JITable function that implements the `size` parameter
    out_dtype = node.outputs[1].type.numpy_dtype
    random_fn_global_env = {
        bcast_fn_name: bcast_fn,
        "out_dtype": out_dtype,
    }

    if tuple_size > 0:
        random_fn_body = dedent(
            f"""
        size = to_fixed_tuple(size, tuple_size)

        data = np.empty(size, dtype=out_dtype)
        for i in np.ndindex(size[:size_dims]):
            data[i] = {bcast_fn_name}({bcast_fn_input_names})

        """
        )
        random_fn_global_env.update(
            {
                "np": np,
                "to_fixed_tuple": numba_ndarray.to_fixed_tuple,
                "tuple_size": tuple_size,
                "size_dims": size_dims,
            }
        )
    else:
        random_fn_body = f"""data = {bcast_fn_name}({bcast_fn_input_names})"""

    sized_fn_src = dedent(
        f"""
def {sized_fn_name}({random_fn_input_names}):
{indent(random_fn_body, " " * 4)}
    return (rng, data)
    """
    )
    random_fn = compile_function_src(
        sized_fn_src, sized_fn_name, {**globals(), **random_fn_global_env}
    )
    random_fn = numba_basic.numba_njit(random_fn)

    return random_fn


@numba_funcify.register(aer.UniformRV)
@numba_funcify.register(aer.TriangularRV)
@numba_funcify.register(aer.BetaRV)
@numba_funcify.register(aer.NormalRV)
@numba_funcify.register(aer.LogNormalRV)
@numba_funcify.register(aer.GammaRV)
@numba_funcify.register(aer.ChiSquareRV)
@numba_funcify.register(aer.ParetoRV)
@numba_funcify.register(aer.GumbelRV)
@numba_funcify.register(aer.ExponentialRV)
@numba_funcify.register(aer.WeibullRV)
@numba_funcify.register(aer.LogisticRV)
@numba_funcify.register(aer.VonMisesRV)
@numba_funcify.register(aer.PoissonRV)
@numba_funcify.register(aer.GeometricRV)
@numba_funcify.register(aer.HyperGeometricRV)
@numba_funcify.register(aer.WaldRV)
@numba_funcify.register(aer.LaplaceRV)
@numba_funcify.register(aer.BinomialRV)
@numba_funcify.register(aer.MultinomialRV)
@numba_funcify.register(aer.RandIntRV)  # only the first two arguments are supported
@numba_funcify.register(aer.ChoiceRV)  # the `p` argument is not supported
@numba_funcify.register(aer.PermutationRV)
def numba_funcify_RandomVariable(op, node, **kwargs):
    name = op.name
    np_random_func = getattr(np.random, name)

    return make_numba_random_fn(node, np_random_func)


def create_numba_random_fn(
    op: Op,
    node: Apply,
    scalar_fn: Callable[[str], str],
    global_env: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Create a vectorized function from a callable that generates the ``str`` function body.

    TODO: This could/should be generalized for other simple function
    construction cases that need unique-ified symbol names.
    """
    np_random_fn_name = f"aesara_random_{get_name_for_object(op.name)}"

    if global_env:
        np_global_env = global_env.copy()
    else:
        np_global_env = {}

    np_global_env["np"] = np
    np_global_env["numba_vectorize"] = numba_basic.numba_vectorize

    unique_names = unique_name_generator(
        [
            np_random_fn_name,
        ]
        + list(np_global_env.keys())
        + [
            "rng",
            "size",
            "dtype",
        ],
        suffix_sep="_",
    )

    np_names = [unique_names(i, force_unique=True) for i in node.inputs[3:]]
    np_input_names = ", ".join(np_names)
    np_random_fn_src = f"""
@numba_vectorize
def {np_random_fn_name}({np_input_names}):
{scalar_fn(*np_names)}
    """
    np_random_fn = compile_function_src(
        np_random_fn_src, np_random_fn_name, {**globals(), **np_global_env}
    )

    return make_numba_random_fn(node, np_random_fn)


@numba_funcify.register(aer.NegBinomialRV)
def numba_funcify_NegBinomialRV(op, node, **kwargs):
    return make_numba_random_fn(node, np.random.negative_binomial)


@numba_funcify.register(aer.CauchyRV)
def numba_funcify_CauchyRV(op, node, **kwargs):
    def body_fn(loc, scale):
        return f"    return ({loc} + np.random.standard_cauchy()) / {scale}"

    return create_numba_random_fn(op, node, body_fn)


@numba_funcify.register(aer.HalfNormalRV)
def numba_funcify_HalfNormalRV(op, node, **kwargs):
    def body_fn(a, b):
        return f"    return {a} + {b} * abs(np.random.normal(0, 1))"

    return create_numba_random_fn(op, node, body_fn)


@numba_funcify.register(aer.BernoulliRV)
def numba_funcify_BernoulliRV(op, node, **kwargs):
    out_dtype = node.outputs[1].type.numpy_dtype

    def body_fn(a):
        return f"""
    if {a} < np.random.uniform(0, 1):
        return direct_cast(0, out_dtype)
    else:
        return direct_cast(1, out_dtype)
        """

    return create_numba_random_fn(
        op,
        node,
        body_fn,
        {"out_dtype": out_dtype, "direct_cast": numba_basic.direct_cast},
    )


@numba_funcify.register(aer.CategoricalRV)
def numba_funcify_CategoricalRV(op, node, **kwargs):
    out_dtype = node.outputs[1].type.numpy_dtype
    size_len = int(get_vector_length(node.inputs[1]))

    @numba_basic.numba_njit
    def categorical_rv(rng, size, dtype, p):
        if not size_len:
            size_tpl = p.shape[:-1]
        else:
            size_tpl = numba_ndarray.to_fixed_tuple(size, size_len)
            p = np.broadcast_to(p, size_tpl + p.shape[-1:])

        unif_samples = np.random.uniform(0, 1, size_tpl)

        res = np.empty(size_tpl, dtype=out_dtype)
        for idx in np.ndindex(*size_tpl):
            res[idx] = np.searchsorted(np.cumsum(p[idx]), unif_samples[idx])

        return (rng, res)

    return categorical_rv


@numba_funcify.register(aer.DirichletRV)
def numba_funcify_DirichletRV(op, node, **kwargs):

    out_dtype = node.outputs[1].type.numpy_dtype
    alphas_ndim = node.inputs[3].type.ndim
    neg_ind_shape_len = -alphas_ndim + 1
    size_len = int(get_vector_length(node.inputs[1]))

    if alphas_ndim > 1:

        @numba_basic.numba_njit
        def dirichlet_rv(rng, size, dtype, alphas):

            if size_len > 0:
                size_tpl = numba_ndarray.to_fixed_tuple(size, size_len)
                if (
                    0 < alphas.ndim - 1 <= len(size_tpl)
                    and size_tpl[neg_ind_shape_len:] != alphas.shape[:-1]
                ):
                    raise ValueError("Parameters shape and size do not match.")
                samples_shape = size_tpl + alphas.shape[-1:]
            else:
                samples_shape = alphas.shape

            res = np.empty(samples_shape, dtype=out_dtype)
            alphas_bcast = np.broadcast_to(alphas, samples_shape)

            for index in np.ndindex(*samples_shape[:-1]):
                res[index] = np.random.dirichlet(alphas_bcast[index])

            return (rng, res)

    else:

        @numba_basic.numba_njit
        def dirichlet_rv(rng, size, dtype, alphas):
            size = numba_ndarray.to_fixed_tuple(size, size_len)
            return (rng, np.random.dirichlet(alphas, size))

    return dirichlet_rv

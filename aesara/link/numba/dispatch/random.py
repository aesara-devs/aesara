from copy import copy
from math import log
from textwrap import dedent, indent
from typing import Callable, Optional

import numba
import numba.np.unsafe.ndarray as numba_ndarray
import numpy as np
from numba import types
from numba.extending import overload, overload_method, register_jitable
from numba.np.random.distributions import random_beta, random_standard_gamma
from numba.np.random.generator_core import next_double
from numba.np.random.generator_methods import check_size, check_types, is_nonelike

import aesara.tensor.random.basic as aer
from aesara.graph.basic import Apply
from aesara.link.numba.dispatch import basic as numba_basic
from aesara.link.numba.dispatch.basic import (
    _numba_funcify,
    create_arg_string,
    get_numba_type,
)
from aesara.link.utils import compile_function_src, unique_name_generator
from aesara.tensor.basic import get_vector_length
from aesara.tensor.random.type import RandomGeneratorType


@get_numba_type.register(RandomGeneratorType)
def get_numba_type_RandomGeneratorType(aesara_type, var, **kwargs):
    return numba.types.npy_rng


@overload(copy)
def copy_NumPyRandomGeneratorType(rng):
    if not isinstance(rng, types.NumPyRandomGeneratorType):
        raise TypeError("`copy` only supports Generators right now")

    def impl(rng):
        # TODO: This seems rather inefficient, but also necessary at this
        # point.  Let's keep an eye out for a better approach.
        with numba.objmode(new_rng=types.npy_rng):
            new_rng = copy(rng)

        return new_rng

    return impl


def make_numba_random_fn(
    node: Apply, sampler_name: str, sampler_fn: Optional[Callable] = None
):
    """Create Numba implementations for Numba-supported `np.random` functions."""
    if not isinstance(node.inputs[0].type, RandomGeneratorType):
        raise TypeError("Numba does not support NumPy `RandomState`s")

    tuple_size = int(get_vector_length(node.inputs[1]))
    size_dims = tuple_size - max(i.ndim for i in node.inputs[3:])

    out_dtype = node.outputs[1].type.numpy_dtype

    if not node.op.inplace:
        copy_rng_stmts = "rng = copy(rng)\n"
    else:
        copy_rng_stmts = ""

    unique_names = unique_name_generator(
        [
            "np",
            "np_random_func",
            "to_fixed_tuple",
            "out_shape",
            "tuple_size",
            "size_dims",
            "rng",
            "size",
            "dtype",
            "i",
            "copy",
        ],
        suffix_sep="_",
    )

    dist_arg_names = [unique_names(i) for i in node.inputs[3:]]
    bcasted_input_stmts = "\n".join(
        [
            f"{name}_bcast = np.broadcast_to({name}, out_shape)"
            if v.type.ndim > 0
            else f"{name}_bcast = {name}"
            for name, v in zip(dist_arg_names, node.inputs[3:])
        ]
    )
    indexed_inputs = create_arg_string(
        [
            f"{name}_bcast[i]" if v.type.ndim > 0 else f"to_scalar({name}_bcast)"
            for name, v in zip(dist_arg_names, node.inputs[3:])
        ]
    )
    random_fn_input_names = ", ".join(["rng", "size", "dtype"] + dist_arg_names)
    input_shape_exprs = create_arg_string(
        [f"np.shape({name})" for name in dist_arg_names]
    )

    out_dtype = node.outputs[1].type.numpy_dtype
    random_fn_global_env = {
        "out_dtype": out_dtype,
        "np": np,
        "to_fixed_tuple": numba_ndarray.to_fixed_tuple,
        "tuple_size": tuple_size,
        "size_dims": size_dims,
        "to_scalar": numba_basic.to_scalar,
        "copy": copy,
    }

    if sampler_fn is not None:
        random_fn_global_env[sampler_name] = sampler_fn
        sampler_fn_expr = f"{sampler_name}(rng, "
        nb_sampler_fn_name = f"{sampler_name}_sampler"
    else:
        sampler_fn_expr = f"rng.{sampler_name}("
        nb_sampler_fn_name = f"{sampler_name}_sampler"

    sampler_fn_src = dedent(
        f"""
    def {nb_sampler_fn_name}({random_fn_input_names}):
        size_tpl = to_fixed_tuple(size, tuple_size)
        out_shape = np.broadcast_shapes(size_tpl, {input_shape_exprs})
{indent(bcasted_input_stmts, " " * 8)}
{indent(copy_rng_stmts, " " * 8)}
        samples = np.empty(out_shape, dtype=out_dtype)
        for i in np.ndindex(out_shape):
            samples[i] = {sampler_fn_expr}{indexed_inputs})

        return (rng, samples)
    """
    ).strip()

    random_fn = compile_function_src(
        sampler_fn_src, nb_sampler_fn_name, {**globals(), **random_fn_global_env}
    )
    random_fn = numba_basic.numba_njit(random_fn)

    return random_fn


@_numba_funcify.register(aer.UniformRV)
@_numba_funcify.register(aer.TriangularRV)
@_numba_funcify.register(aer.BetaRV)
@_numba_funcify.register(aer.NormalRV)
@_numba_funcify.register(aer.LogNormalRV)
@_numba_funcify.register(aer.GammaRV)
@_numba_funcify.register(aer.ChiSquareRV)
@_numba_funcify.register(aer.ParetoRV)
@_numba_funcify.register(aer.GumbelRV)
@_numba_funcify.register(aer.ExponentialRV)
@_numba_funcify.register(aer.WeibullRV)
@_numba_funcify.register(aer.LogisticRV)
@_numba_funcify.register(aer.VonMisesRV)
@_numba_funcify.register(aer.PoissonRV)
@_numba_funcify.register(aer.GeometricRV)
# @_numba_funcify.register(aer.HyperGeometricRV)
@_numba_funcify.register(aer.WaldRV)
@_numba_funcify.register(aer.LaplaceRV)
# @_numba_funcify.register(aer.BinomialRV)
@_numba_funcify.register(aer.MultinomialRV)
@_numba_funcify.register(aer.ChoiceRV)  # the `p` argument is not supported
@_numba_funcify.register(aer.PermutationRV)
def numba_funcify_RandomVariable(op, node, **kwargs):
    return make_numba_random_fn(node, op.name)


@_numba_funcify.register(aer.NegBinomialRV)
def numba_funcify_NegBinomialRV(op, node, **kwargs):
    return make_numba_random_fn(node, "negative_binomial")


def gamma_scalar_fn(rng, shape, scale):
    return rng.gamma(shape, scale)


@_numba_funcify.register(aer.GammaRV)
def numba_funcify_GammaRV(op, node, **kwargs):
    scalar_fn = numba_basic.numba_njit(gamma_scalar_fn)
    return make_numba_random_fn(node, "gamma", scalar_fn)


def cauchy_scalar_fn(rng, loc, scale):
    return loc + rng.standard_cauchy() * scale


@_numba_funcify.register(aer.CauchyRV)
def numba_funcify_CauchyRV(op, node, **kwargs):
    scalar_fn = numba_basic.numba_njit(cauchy_scalar_fn)
    return make_numba_random_fn(node, "cauchy", scalar_fn)


def pareto_scalar_fn(rng, b, scale):
    return rng.pareto(b) / scale


@_numba_funcify.register(aer.ParetoRV)
def numba_funcify_ParetoRV(op, node, **kwargs):
    scalar_fn = numba_basic.numba_njit(pareto_scalar_fn)
    return make_numba_random_fn(node, "pareto", scalar_fn)


def halfnormal_scalar_fn(rng, loc, scale):
    return loc + abs(rng.standard_normal()) * scale


@_numba_funcify.register(aer.HalfNormalRV)
def numba_funcify_HalfNormalRV(op, node, **kwargs):
    scalar_fn = numba_basic.numba_njit(halfnormal_scalar_fn)
    return make_numba_random_fn(node, "halfnormal", scalar_fn)


@_numba_funcify.register(aer.BernoulliRV)
def numba_funcify_BernoulliRV(op, node, **kwargs):
    out_dtype = node.outputs[1].type.numpy_dtype

    @numba_basic.numba_njit
    def scalar_fn(rng, a):
        if a < rng.uniform(0, 1):
            return numba_basic.direct_cast(0, out_dtype)
        else:
            return numba_basic.direct_cast(1, out_dtype)

    return make_numba_random_fn(node, "bernoulli", scalar_fn)


@_numba_funcify.register(aer.CategoricalRV)
def numba_funcify_CategoricalRV(op, node, **kwargs):
    out_dtype = node.outputs[1].type.numpy_dtype
    size_len = int(get_vector_length(node.inputs[1]))
    inplace = op.inplace

    @numba_basic.numba_njit
    def categorical_rv(rng, size, dtype, p):
        if not inplace:
            rng = copy(rng)

        if not size_len:
            size_tpl = p.shape[:-1]
        else:
            size_tpl = numba_ndarray.to_fixed_tuple(size, size_len)
            p = np.broadcast_to(p, size_tpl + p.shape[-1:])

        unif_samples = np.asarray(rng.uniform(0, 1, size_tpl))

        res = np.empty(size_tpl, dtype=out_dtype)
        for idx in np.ndindex(*size_tpl):
            res[idx] = np.searchsorted(np.cumsum(p[idx]), unif_samples[idx])

        return (rng, res)

    return categorical_rv


@_numba_funcify.register(aer.DirichletRV)
def numba_funcify_DirichletRV(op, node, **kwargs):
    out_dtype = node.outputs[1].type.numpy_dtype
    alphas_ndim = node.inputs[3].type.ndim
    neg_ind_shape_len = -alphas_ndim + 1
    size_len = int(get_vector_length(node.inputs[1]))
    inplace = op.inplace

    if alphas_ndim > 1:

        @numba_basic.numba_njit
        def dirichlet_rv(rng, size, dtype, alphas):
            if not inplace:
                rng = copy(rng)

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
                res[index] = rng.dirichlet(alphas_bcast[index])

            return (rng, res)

    else:

        @numba_basic.numba_njit
        def dirichlet_rv(rng, size, dtype, alphas):
            if not inplace:
                rng = copy(rng)

            size = numba_ndarray.to_fixed_tuple(size, size_len)
            return (rng, rng.dirichlet(alphas, size))

    return dirichlet_rv


@register_jitable
def random_dirichlet(bitgen, alpha, size):
    """
    This implementation is straight from ``numpy/random/_generator.pyx``.
    """

    k = len(alpha)
    alpha_arr = np.asarray(alpha, dtype=np.float64)

    if np.any(np.less_equal(alpha_arr, 0)):
        raise ValueError("alpha <= 0")

    shape = size + (k,)

    diric = np.zeros(shape, np.float64)

    i = 0
    totsize = diric.size

    if (k > 0) and (alpha_arr.max() < 0.1):
        alpha_csum_arr = np.empty_like(alpha_arr)
        csum = 0.0
        for j in range(k - 1, -1, -1):
            csum += alpha_arr[j]
            alpha_csum_arr[j] = csum

        while i < totsize:
            acc = 1.0
            for j in range(k - 1):
                v = random_beta(bitgen, alpha_arr[j], alpha_csum_arr[j + 1])
                diric[i + j] = acc * v
                acc *= 1.0 - v
            diric[i + k - 1] = acc
            i = i + k

    else:
        while i < totsize:
            acc = 0.0
            for j in range(k):
                diric[i + j] = random_standard_gamma(bitgen, alpha_arr[j])
                acc = acc + diric[i + j]
            invacc = 1.0 / acc
            for j in range(k):
                diric[i + j] = diric[i + j] * invacc
            i = i + k

    return diric


@overload_method(types.NumPyRandomGeneratorType, "dirichlet")
def NumPyRandomGeneratorType_dirichlet(inst, alphas, size=None):
    check_types(alphas, [types.Array, types.List], "alphas")

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):

        def impl(inst, alphas, size=None):
            return random_dirichlet(inst.bit_generator, alphas, ())

    elif isinstance(size, (int, types.Integer)):

        def impl(inst, alphas, size=None):
            return random_dirichlet(inst.bit_generator, alphas, (size,))

    else:
        check_size(size)

        def impl(inst, alphas, size=None):
            return random_dirichlet(inst.bit_generator, alphas, size)

    return impl


@register_jitable
def random_gumbel(bitgen, loc, scale):
    """
    This implementation is adapted from ``numpy/random/src/distributions/distributions.c``.
    """
    while True:
        u = 1.0 - next_double(bitgen)
        if u < 1.0:
            return loc - scale * log(-log(u))


@overload_method(types.NumPyRandomGeneratorType, "gumbel")
def NumPyRandomGeneratorType_gumbel(inst, loc=0.0, scale=1.0, size=None):
    check_types(loc, [types.Float, types.Integer, int, float], "loc")
    check_types(scale, [types.Float, types.Integer, int, float], "scale")

    if isinstance(size, types.Omitted):
        size = size.value

    if is_nonelike(size):

        def impl(inst, loc=0.0, scale=1.0, size=None):
            return random_gumbel(inst.bit_generator, loc, scale)

    else:
        check_size(size)

        def impl(inst, loc=0.0, scale=1.0, size=None):
            out = np.empty(size)
            for i in np.ndindex(size):
                out[i] = random_gumbel(inst.bit_generator, loc, scale)
            return out

    return impl

import re

import numpy as np
import pytest
import scipy.stats as stats

import aesara
import aesara.tensor as at
import aesara.tensor.random as aer
from aesara.compile.function import function
from aesara.compile.sharedvalue import SharedVariable, shared
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.basic import RandomVariable
from aesara.tensor.random.utils import RandomStream
from tests.link.jax.test_basic import compare_jax_and_py, jax_mode, set_test_value


jax = pytest.importorskip("jax")


def test_random_RandomStream():
    """Two successive calls of a compiled graph using `RandomStream` should
    return different values.

    """
    srng = RandomStream(seed=123)
    out = srng.normal() - srng.normal()

    with pytest.warns(
        UserWarning,
        match=r"The RandomType SharedVariables \[.+\] will not be used",
    ):
        fn = function([], out, mode=jax_mode)
    jax_res_1 = fn()
    jax_res_2 = fn()

    assert not np.array_equal(jax_res_1, jax_res_2)


@pytest.mark.parametrize("rng_ctor", (np.random.RandomState, np.random.default_rng))
def test_random_updates(rng_ctor):
    original_value = rng_ctor(seed=98)
    rng = shared(original_value, name="original_rng", borrow=False)
    next_rng, x = at.random.normal(name="x", rng=rng).owner.outputs

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The RandomType SharedVariables [original_rng] will not be used"
        ),
    ):
        f = aesara.function([], [x], updates={rng: next_rng}, mode=jax_mode)
    assert f() != f()

    # Check that original rng variable content was not overwritten when calling jax_typify
    assert all(
        a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
        for a, b in zip(rng.get_value().__getstate__(), original_value.__getstate__())
    )


@pytest.mark.parametrize(
    "rv_op, dist_params, base_size, cdf_name, params_conv, n_samples",
    [
        (
            aer.beta,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "beta",
            lambda *args: args,
            None,
        ),
        (
            aer.cauchy,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "cauchy",
            lambda *args: args,
            None,
        ),
        (
            aer.chisquare,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                )
            ],
            (2,),
            "chi2",
            lambda *args: args,
            50_000,
        ),
        (
            aer.exponential,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            (2,),
            "expon",
            lambda *args: (0, args[0]),
            None,
        ),
        (
            aer.gamma,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "gamma",
            lambda a, b: (a, 0.0, b),
            None,
        ),
        (
            aer.gumbel,
            [
                set_test_value(
                    at.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "gumbel_r",
            lambda *args: args,
            None,
        ),
        (
            aer.laplace,
            [
                set_test_value(at.dvector(), np.array([1.0, 2.0], dtype=np.float64)),
                set_test_value(at.dscalar(), np.array(1.0, dtype=np.float64)),
            ],
            (2,),
            "laplace",
            lambda *args: args,
            None,
        ),
        (
            aer.logistic,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "logistic",
            lambda *args: args,
            None,
        ),
        (
            aer.lognormal,
            [
                set_test_value(
                    at.lvector(),
                    np.array([0, 0], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "lognorm",
            lambda *args: args,
            None,
        ),
        (
            aer.normal,
            [
                set_test_value(
                    at.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "norm",
            lambda *args: args,
            None,
        ),
        (
            aer.pareto,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                )
            ],
            (2,),
            "pareto",
            lambda *args: args,
            None,
        ),
        (
            aer.poisson,
            [
                set_test_value(
                    at.dvector(),
                    np.array([100000.0, 200000.0], dtype=np.float64),
                ),
            ],
            (2,),
            "poisson",
            lambda *args: args,
            None,
        ),
        (
            aer.randint,
            [
                set_test_value(
                    at.lscalar(),
                    np.array(0, dtype=np.int64),
                ),
                set_test_value(  # high-value necessary since test on cdf
                    at.lscalar(),
                    np.array(1000, dtype=np.int64),
                ),
            ],
            (),
            "randint",
            lambda *args: args,
            None,
        ),
        (
            aer.standard_normal,
            [],
            (2,),
            "norm",
            lambda *args: args,
            None,
        ),
        (
            aer.t,
            [
                set_test_value(
                    at.dscalar(),
                    np.array(2.0, dtype=np.float64),
                ),
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "t",
            lambda *args: args,
            50_000,
        ),
        (
            aer.uniform,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1000.0, dtype=np.float64),
                ),
            ],
            (2,),
            "uniform",
            lambda *args: args,
            50_000,
        ),
        (
            aer.halfnormal,
            [
                set_test_value(
                    at.dvector(),
                    np.array([-1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1000.0, dtype=np.float64),
                ),
            ],
            (2,),
            "halfnorm",
            lambda *args: args,
            None,
        ),
        (
            aer.halfcauchy,
            [
                set_test_value(
                    at.dvector(),
                    np.array([-1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1000.0, dtype=np.float64),
                ),
            ],
            (2,),
            "halfcauchy",
            lambda *args: args,
            None,
        ),
        (
            aer.wald,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0], dtype=np.float64),
                ),
            ],
            (),
            "invgauss",
            lambda *args: args,
            None,
        ),
        (
            aer.gengamma,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 4.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dvector(),
                    np.array([2.0, 4.0], dtype=np.float64),
                ),
            ],
            (2,),
            "gengamma",
            lambda alpha, d, p: (alpha / p, p, 0, d),
            50_000,
        ),
    ],
)
def test_random_RandomVariable(
    rv_op, dist_params, base_size, cdf_name, params_conv, n_samples
):
    """The JAX samplers are not one-to-one with NumPy samplers so we
    need to use a statistical test to make sure that the transpilation
    is correct.

    Parameters
    ----------
    rv_op
        The transpiled `RandomVariable` `Op`.
    dist_params
        The parameters passed to the op.

    """
    if n_samples is None:
        n_samples = 10_000

    rng = shared(np.random.RandomState(29402))
    g = rv_op(*dist_params, size=(n_samples,) + base_size, rng=rng)
    g_fn = function(dist_params, g, mode=jax_mode)
    samples = g_fn(
        *[
            i.tag.test_value
            for i in g_fn.maker.fgraph.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ]
    )

    bcast_dist_args = np.broadcast_arrays(*[i.tag.test_value for i in dist_params])

    for idx in np.ndindex(*base_size):
        cdf_params = params_conv(*tuple(arg[idx] for arg in bcast_dist_args))
        test_res = stats.cramervonmises(
            samples[(Ellipsis,) + idx], cdf_name, args=cdf_params
        )
        assert test_res.pvalue > 0.1


@pytest.mark.parametrize("size", [(), (4,)])
def test_random_bernoulli(size):
    rng = shared(np.random.RandomState(123))
    g = at.random.bernoulli(0.5, size=(1000,) + size, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


@pytest.mark.parametrize(
    "p, size",
    [
        (0.6, ()),
        (0.2, (4,)),
    ],
)
def test_random_geometric(p, size):
    rng = shared(np.random.RandomState(123))
    g = at.random.geometric(p, size=(1000,) + size, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(), 1 / p, atol=0.1)


def test_random_mvnormal():
    rng = shared(np.random.RandomState(123))

    mu = np.ones(4)
    cov = np.eye(4)
    g = at.random.multivariate_normal(mu, cov, size=(10000,), rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), mu, atol=0.1)


@pytest.mark.parametrize(
    "parameter, size",
    [
        (np.ones(4), ()),
        (np.ones(4), (2, 4)),
    ],
)
def test_random_dirichlet(parameter, size):
    rng = shared(np.random.RandomState(123))
    g = at.random.dirichlet(parameter, size=(1000,) + size, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


@pytest.mark.parametrize(
    "shape, scale",
    [(3, 3), (2, 1), (2, 5)],
)
def test_random_invgamma(shape, scale):
    rng = shared(np.random.RandomState(123))
    g = at.random.invgamma(shape, scale, size=(100000,), rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    # mean = scale / (shape - 1) only exists for shape > 1
    np.testing.assert_allclose(samples.mean(), scale / (shape - 1), rtol=1e-01)


def test_random_choice():
    # Elements are picked at equal frequency
    num_samples = 10000
    rng = shared(np.random.RandomState(123))
    g = at.random.choice(np.arange(4), size=num_samples, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(np.sum(samples == 3) / num_samples, 0.25, 2)

    # `replace=False` produces unique results
    rng = shared(np.random.RandomState(123))
    g = at.random.choice(np.arange(100), replace=False, size=99, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    assert len(np.unique(samples)) == 99

    # We can pass an array with probabilities
    rng = shared(np.random.RandomState(123))
    g = at.random.choice(np.arange(3), p=np.array([1.0, 0.0, 0.0]), size=10, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples, np.zeros(10))


def test_random_categorical():
    rng = shared(np.random.RandomState(123))
    g = at.random.categorical(0.25 * np.ones(4), size=(10000, 4), rng=rng)
    g_fn = function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 6 / 4, 1)


def test_random_permutation():
    array = np.arange(4)
    rng = shared(np.random.RandomState(123))
    g = at.random.permutation(array, rng=rng)
    g_fn = function([], g, mode=jax_mode)
    permuted = g_fn()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(array, permuted)


def test_random_unimplemented():
    """Compiling a graph with a non-supported `RandomVariable` should
    raise an error.

    """

    class NonExistentRV(RandomVariable):
        name = "non-existent"
        ndim_supp = 0
        ndims_params = []
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    nonexistentrv = NonExistentRV()
    rng = shared(np.random.RandomState(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)

    with pytest.raises(NotImplementedError):
        compare_jax_and_py(fgraph, [])


def test_random_custom_implementation():
    """We can register a JAX implementation for user-defined `RandomVariable`s"""

    class CustomRV(RandomVariable):
        name = "non-existent"
        ndim_supp = 0
        ndims_params = []
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    from aesara.link.jax.dispatch.random import jax_sample_fn

    @jax_sample_fn.register(CustomRV)
    def jax_sample_fn_custom(op):
        def sample_fn(rng, size, dtype, *parameters):
            return (rng, 0)

        return sample_fn

    nonexistentrv = CustomRV()
    rng = shared(np.random.RandomState(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)
    compare_jax_and_py(fgraph, [])


def test_random_concrete_shape():
    """JAX should compile when a `RandomVariable` is passed a concrete shape.

    There are three quantities that JAX considers as concrete:
    1. Constants known at compile time;
    2. The shape of an array.
    3. `static_argnums` parameters
    This test makes sure that graphs with `RandomVariable`s compile when the
    `size` parameter satisfies either of these criteria.

    """
    rng = shared(np.random.RandomState(123))
    x_at = at.dmatrix()
    out = at.random.normal(0, 1, size=x_at.shape, rng=rng)
    jax_fn = function([x_at], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (2, 3)


def test_random_concrete_shape_subtensor():
    """JAX should compile when a concrete value is passed for the `size` parameter.

    This test ensures that the `DimShuffle` `Op` used by Aesara to turn scalar
    inputs into 1d vectors is replaced by an `Op` that turns concrete scalar
    inputs into tuples of concrete values using the `jax_size_parameter_as_tuple`
    rewrite.

    JAX does not accept scalars as `size` or `shape` arguments, so this is a
    slight improvement over their API.

    """
    rng = shared(np.random.RandomState(123))
    x_at = at.dmatrix()
    out = at.random.normal(0, 1, size=x_at.shape[1], rng=rng)
    jax_fn = function([x_at], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (3,)


def test_random_concrete_shape_subtensor_tuple():
    """JAX should compile when a tuple of concrete values is passed for the `size` parameter.

    This test ensures that the `MakeVector` `Op` used by Aesara to turn tuple
    inputs into 1d vectors is replaced by an `Op` that turns a tuple of concrete
    scalar inputs into tuples of concrete values using the
    `jax_size_parameter_as_tuple` rewrite.

    """
    rng = shared(np.random.RandomState(123))
    x_at = at.dmatrix()
    out = at.random.normal(0, 1, size=(x_at.shape[0],), rng=rng)
    jax_fn = function([x_at], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (2,)


def test_random_concrete_shape_graph_input():
    """JAX cannot JIT-compile random variables whose `size` argument is not static."""
    rng = shared(np.random.RandomState(123))
    size_at = at.scalar()
    out = at.random.normal(0, 1, size=size_at, rng=rng)
    with pytest.raises(NotImplementedError, match=r".* concrete values .*"):
        function([size_at], out, mode=jax_mode)

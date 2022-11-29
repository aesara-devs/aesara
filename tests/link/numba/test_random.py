import contextlib

import numpy as np
import pytest
import scipy.stats as stats

import aesara.tensor as at
import aesara.tensor.random.basic as aer
from aesara import shared
from aesara.compile.function import function
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    eval_python_only,
    numba_mode,
    set_test_value,
)


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "rv_op, dist_args, size",
    [
        (
            aer.normal,
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
            at.as_tensor([3, 2]),
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
                    np.array(3.0, dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
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
            at.as_tensor([3, 2]),
        ),
        (
            aer.triangular,
            [
                set_test_value(
                    at.dscalar(),
                    np.array(-5.0, dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(5.0, dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
        ),
        (
            aer.lognormal,
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
            at.as_tensor([3, 2]),
        ),
        (
            aer.exponential,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
        ),
        (
            aer.weibull,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
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
            at.as_tensor([3, 2]),
        ),
        (
            aer.geometric,
            [
                set_test_value(
                    at.dvector(),
                    np.array([0.3, 0.4], dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
        ),
        (
            # TODO FIXME: This works, but uses object-mode.
            aer.hypergeometric,
            [
                set_test_value(
                    at.lscalar(),
                    np.array(7, dtype=np.int64),
                ),
                set_test_value(
                    at.lscalar(),
                    np.array(8, dtype=np.int64),
                ),
                set_test_value(
                    at.lscalar(),
                    np.array(15, dtype=np.int64),
                ),
            ],
            at.as_tensor([3, 2]),
        ),
        (
            aer.wald,
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
            at.as_tensor([3, 2]),
        ),
        (
            aer.laplace,
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
            at.as_tensor([3, 2]),
        ),
        (
            # TODO FIXME: This works, but uses object-mode.
            aer.binomial,
            [
                set_test_value(
                    at.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(0.9, dtype=np.float64),
                ),
            ],
            at.as_tensor([3, 2]),
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
            at.as_tensor(tuple(set_test_value(at.lscalar(), v) for v in [3, 2])),
        ),
        (
            aer.poisson,
            [
                set_test_value(
                    at.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            aer.bernoulli,
            [
                set_test_value(
                    at.dvector(),
                    np.array([0.1, 0.9], dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            aer.integers,
            [
                set_test_value(
                    at.lscalar(),
                    np.array(0, dtype=np.int64),
                ),
                set_test_value(
                    at.lscalar(),
                    np.array(5, dtype=np.int64),
                ),
            ],
            at.as_tensor([3, 2]),
        ),
        (
            # TODO FIXME: This works, but uses object-mode.
            aer.multivariate_normal,
            [
                set_test_value(
                    at.dmatrix(),
                    np.array([[1, 2], [3, 4]], dtype=np.float64),
                ),
                set_test_value(
                    at.tensor("float64", shape=(1, None, None)),
                    np.eye(2)[None, ...],
                ),
            ],
            at.as_tensor(tuple(set_test_value(at.lscalar(), v) for v in [4, 3, 2])),
        ),
    ],
    ids=str,
)
def test_aligned_RandomVariable(rv_op, dist_args, size):
    """Tests for Numba samplers that are one-to-one with Aesara's/NumPy's samplers."""
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=size, rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "rv_op, dist_args, base_size, cdf_name, params_conv",
    [
        (
            aer.pareto,
            [
                set_test_value(
                    at.dvector(),
                    np.array([2.0, 3.0], dtype=np.float64),
                ),
            ],
            (2,),
            "lomax",
            lambda b, scale=1.0: (b, 0.0, scale),
        ),
        (
            aer.halfnormal,
            [
                set_test_value(
                    at.lvector(),
                    np.array([0, 1], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "halfnorm",
            lambda *args: args,
        ),
        (
            aer.beta,
            [
                set_test_value(
                    at.dvector(),
                    np.array([0.5, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(0.5, dtype=np.float64),
                ),
            ],
            (2,),
            "beta",
            lambda *args: args,
        ),
        (
            aer.cauchy,
            [
                set_test_value(
                    at.dvector(),
                    np.array([0.0, 10.0], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "cauchy",
            lambda *args: args,
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
        ),
        (
            aer.gumbel,
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
            "gumbel_r",
            lambda *args: args,
        ),
        (
            aer.negative_binomial,
            [
                set_test_value(
                    at.lvector(),
                    np.array([10, 20], dtype=np.int64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(0.09, dtype=np.float64),
                ),
            ],
            (2,),
            "nbinom",
            lambda *args: args,
        ),
        pytest.param(
            aer.vonmises,
            [
                set_test_value(
                    at.dvector(),
                    np.array([-0.5, 0.5], dtype=np.float64),
                ),
                set_test_value(
                    at.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "vonmises_line",
            lambda mu, kappa: (kappa, mu),
            marks=pytest.mark.xfail(
                reason=(
                    "Numba's parameterization of `vonmises` does not match NumPy's."
                    "See https://github.com/numba/numba/issues/7886"
                )
            ),
        ),
    ],
)
def test_unaligned_RandomVariable(rv_op, dist_args, base_size, cdf_name, params_conv):
    """Tests for Numba samplers that are not one-to-one with Aesara's/NumPy's samplers.

    TODO FIXME: The reason why we can't directly compare the Aesara samples
    with Numba's is that Aesara actually uses SciPy instead of NumPy to sample
    them.
    """
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=(2000,) + base_size, rng=rng)
    g_fn = function(dist_args, g, mode=numba_mode)
    arg_vals = [
        i.tag.test_value
        for i in g_fn.maker.fgraph.inputs
        if not isinstance(i, (SharedVariable, Constant))
    ]
    samples = g_fn(*arg_vals)

    bcast_dist_args = np.broadcast_arrays(*[i.tag.test_value for i in dist_args])

    for idx in np.ndindex(*base_size):
        cdf_params = params_conv(*tuple(arg[idx] for arg in bcast_dist_args))
        test_res = stats.cramervonmises(
            samples[(Ellipsis,) + idx], cdf_name, args=cdf_params
        )
        assert test_res.pvalue > 0.1


@pytest.mark.parametrize(
    "dist_args, size, cm",
    [
        pytest.param(
            [
                set_test_value(
                    at.dvector(),
                    np.array([100000, 1, 1], dtype=np.float64),
                ),
            ],
            None,
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    at.dmatrix(),
                    np.array(
                        [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (10, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    at.dmatrix(),
                    np.array(
                        [[100000, 1, 1]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (5, 4, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    at.dmatrix(),
                    np.array(
                        [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (10, 4),
            pytest.raises(
                ValueError, match="objects cannot be broadcast to a single shape"
            ),
        ),
    ],
)
def test_CategoricalRV(dist_args, size, cm):
    rng = shared(np.random.default_rng(29402))
    g = aer.categorical(*dist_args, size=size, rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "a, size, cm",
    [
        pytest.param(
            set_test_value(
                at.dvector(),
                np.array([100000, 1, 1], dtype=np.float64),
            ),
            None,
            contextlib.suppress(),
        ),
        pytest.param(
            set_test_value(
                at.dmatrix(),
                np.array(
                    [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                    dtype=np.float64,
                ),
            ),
            (10, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            set_test_value(
                at.dmatrix(),
                np.array(
                    [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                    dtype=np.float64,
                ),
            ),
            (10, 4),
            pytest.raises(ValueError, match="Parameters shape.*"),
        ),
    ],
)
def test_DirichletRV(a, size, cm):
    rng = shared(np.random.default_rng(29402))
    g = aer.dirichlet(a, size=size, rng=rng)
    g_fn = function([a], g, mode=numba_mode)

    with cm:
        a_val = a.tag.test_value

        # For coverage purposes only...
        eval_python_only([a], [g], [a_val])

        all_samples = []
        for i in range(1000):
            samples = g_fn(a_val)
            all_samples.append(samples)

        exp_res = a_val / a_val.sum(-1)
        res = np.mean(all_samples, axis=tuple(range(0, a_val.ndim - 1)))
        assert np.allclose(res, exp_res, atol=1e-4)


def test_updates():
    rng = shared(np.random.default_rng(1))
    rng_new = shared(np.random.default_rng(2))

    x = at.random.normal(size=10, rng=rng)
    x_fn = function([], x, updates={rng: rng_new}, mode=numba_mode)

    ref = np.random.default_rng(1).normal(size=10)
    assert np.allclose(x_fn(), ref)

    ref = np.random.default_rng(2).normal(size=10)
    assert np.allclose(x_fn(), ref)


def test_RandomState_error():
    rng = shared(np.random.RandomState(29402))
    g = aer.normal(rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    with pytest.raises(TypeError):
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )

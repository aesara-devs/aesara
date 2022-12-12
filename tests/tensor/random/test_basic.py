import pickle
import re
from copy import copy

import numpy as np
import pytest
import scipy.stats as stats

import aesara.tensor as at
from aesara import function, shared
from aesara.compile.mode import Mode
from aesara.compile.sharedvalue import SharedVariable
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.tensor.random.basic import (
    bernoulli,
    beta,
    betabinom,
    binomial,
    broadcast_shapes,
    categorical,
    cauchy,
    chisquare,
    choice,
    dirichlet,
    exponential,
    gamma,
    gengamma,
    geometric,
    gumbel,
    halfcauchy,
    halfnormal,
    hypergeometric,
    integers,
    invgamma,
    laplace,
    logistic,
    lognormal,
    multinomial,
    multivariate_normal,
    nbinom,
    normal,
    pareto,
    permutation,
    poisson,
    randint,
    random,
    standard_normal,
    t,
    triangular,
    truncexpon,
    uniform,
    vonmises,
    wald,
    weibull,
)
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.type import iscalar, scalar, tensor
from tests.unittest_tools import create_aesara_param


rewrites_query = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
py_mode = Mode("py", rewrites_query)


def fixed_scipy_rvs(rvs_name):
    """Create a SciPy sampling function compatible with the `test_fn` argument of `compare_sample_values`."""

    def _rvs(*args, size=None, **kwargs):
        res = getattr(stats, rvs_name).rvs(*args, size=size, **kwargs)
        res = np.broadcast_to(
            res,
            size
            if size is not None
            else broadcast_shapes(*[np.shape(a) for a in args]),
        )
        return res

    return _rvs


def compare_sample_values(rv, *params, rng=None, test_fn=None, **kwargs):
    """Test for equivalence between `RandomVariable` and NumPy/other samples.

    An equivalently named method on a NumPy RNG object will be used, unless
    `test_fn` is specified.

    """
    if rng is None:
        rng = np.random.default_rng()

    if test_fn is None:
        name = getattr(rv, "name", None)

        if name is None:
            name = rv.__name__

        def test_fn(*args, random_state=None, **kwargs):
            return getattr(random_state, name)(*args, **kwargs)

    param_vals = [get_test_value(p) if isinstance(p, Variable) else p for p in params]
    kwargs_vals = {
        k: get_test_value(v) if isinstance(v, Variable) else v
        for k, v in kwargs.items()
    }

    at_rng = shared(rng, borrow=True)

    numpy_res = np.asarray(test_fn(*param_vals, random_state=copy(rng), **kwargs_vals))

    aesara_res = rv(*params, rng=at_rng, **kwargs)

    assert aesara_res.type.numpy_dtype.kind == numpy_res.dtype.kind

    numpy_shape = np.shape(numpy_res)
    numpy_bcast = [s == 1 for s in numpy_shape]
    np.testing.assert_array_equal(aesara_res.type.broadcastable, numpy_bcast)

    fn_inputs = [
        i
        for i in graph_inputs([aesara_res])
        if not isinstance(i, (Constant, SharedVariable))
    ]
    aesara_fn = function(fn_inputs, aesara_res, mode=py_mode)

    aesara_res_val = aesara_fn()

    assert aesara_res_val.flags.writeable

    np.testing.assert_array_equal(aesara_res_val.shape, numpy_res.shape)

    np.testing.assert_allclose(aesara_res_val, numpy_res)


@pytest.mark.parametrize(
    "u, l, size",
    [
        (np.array(10, dtype=config.floatX), np.array(20, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), np.array(20, dtype=config.floatX), []),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(20, dtype=config.floatX),
            None,
        ),
    ],
)
def test_uniform_samples(u, l, size):
    compare_sample_values(uniform, u, l, size=size)


def test_uniform_default_args():
    compare_sample_values(uniform)


@pytest.mark.parametrize(
    "left, mode, right, size",
    [
        (
            np.array(10, dtype=config.floatX),
            np.array(12, dtype=config.floatX),
            np.array(20, dtype=config.floatX),
            None,
        ),
        (
            np.array(10, dtype=config.floatX),
            np.array(12, dtype=config.floatX),
            np.array(20, dtype=config.floatX),
            [],
        ),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(12, dtype=config.floatX),
            np.array(20, dtype=config.floatX),
            None,
        ),
    ],
)
def test_triangular_samples(left, mode, right, size):
    compare_sample_values(triangular, left, mode, right, size=size)


@pytest.mark.parametrize(
    "a, b, size",
    [
        (np.array(0.5, dtype=config.floatX), np.array(0.5, dtype=config.floatX), None),
        (np.array(0.5, dtype=config.floatX), np.array(0.5, dtype=config.floatX), []),
        (
            np.full((1, 2), 0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_beta_samples(a, b, size):
    compare_sample_values(beta, a, b, size=size)


M_at = iscalar("M")
M_at.tag.test_value = 3
sd_at = scalar("sd")
sd_at.tag.test_value = np.array(1.0, dtype=config.floatX)


@pytest.mark.parametrize(
    "M, sd, size",
    [
        (at.as_tensor_variable(np.array(1.0, dtype=config.floatX)), sd_at, ()),
        (
            at.as_tensor_variable(np.array(1.0, dtype=config.floatX)),
            sd_at,
            (M_at,),
        ),
        (
            at.as_tensor_variable(np.array(1.0, dtype=config.floatX)),
            sd_at,
            (2, M_at),
        ),
        (at.zeros((M_at,)), sd_at, ()),
        (at.zeros((M_at,)), sd_at, (M_at,)),
        (at.zeros((M_at,)), sd_at, (2, M_at)),
        (at.zeros((M_at,)), at.ones((M_at,)), ()),
        (at.zeros((M_at,)), at.ones((M_at,)), (2, M_at)),
        (
            create_aesara_param(
                np.array([[-1, 20], [300, -4000]], dtype=config.floatX)
            ),
            create_aesara_param(np.array([[1e-6, 2e-6]], dtype=config.floatX)),
            (3, 2, 2),
        ),
        (
            create_aesara_param(np.array([1], dtype=config.floatX)),
            create_aesara_param(np.array([10], dtype=config.floatX)),
            (1, 2),
        ),
    ],
)
def test_normal_infer_shape(M, sd, size):
    rv = normal(M, sd, size=size)
    rv_shape = list(normal._infer_shape(size or (), [M, sd], None))

    all_args = (M, sd) + size
    fn_inputs = [
        i
        for i in graph_inputs([a for a in all_args if isinstance(a, Variable)])
        if not isinstance(i, (Constant, SharedVariable))
    ]
    aesara_fn = function(
        fn_inputs, [at.as_tensor(o) for o in rv_shape + [rv]], mode=py_mode
    )

    *rv_shape_val, rv_val = aesara_fn(
        *[
            i.tag.test_value
            for i in fn_inputs
            if not isinstance(i, (SharedVariable, Constant))
        ]
    )

    assert tuple(rv_shape_val) == tuple(rv_val.shape)


@config.change_flags(compute_test_value="raise")
def test_normal_ShapeFeature():
    M_at = iscalar("M")
    M_at.tag.test_value = 3
    sd_at = scalar("sd")
    sd_at.tag.test_value = np.array(1.0, dtype=config.floatX)

    d_rv = normal(at.ones((M_at,)), sd_at, size=(2, M_at))
    d_rv.tag.test_value

    fg = FunctionGraph(
        [i for i in graph_inputs([d_rv]) if not isinstance(i, Constant)],
        [d_rv],
        clone=False,
        features=[ShapeFeature()],
    )
    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == get_test_value(d_rv).shape[0]
    assert get_test_value(s2) == get_test_value(d_rv).shape[1]


@pytest.mark.parametrize(
    "mean, sigma, size",
    [
        (np.array(100, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), None),
        (np.array(100, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), []),
        (
            np.full((1, 2), 100, dtype=config.floatX),
            np.array(1e-2, dtype=config.floatX),
            None,
        ),
    ],
)
def test_normal_samples(mean, sigma, size):
    compare_sample_values(normal, mean, sigma, size=size)


def test_normal_default_args():
    compare_sample_values(standard_normal)


def test_random_default_args():
    compare_sample_values(random)


@pytest.mark.parametrize(
    "mean, sigma, size",
    [
        (np.array(100, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), None),
        (np.array(100, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), []),
        (
            np.full((1, 2), 100, dtype=config.floatX),
            np.array(1e-2, dtype=config.floatX),
            None,
        ),
    ],
)
def test_halfnormal_samples(mean, sigma, size):
    compare_sample_values(
        halfnormal, mean, sigma, size=size, test_fn=fixed_scipy_rvs("halfnorm")
    )


@pytest.mark.parametrize(
    "mean, sigma, size",
    [
        (np.array(10, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), np.array(1e-2, dtype=config.floatX), []),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(1e-2, dtype=config.floatX),
            None,
        ),
    ],
)
def test_lognormal_samples(mean, sigma, size):
    compare_sample_values(lognormal, mean, sigma, size=size)


@pytest.mark.parametrize(
    "a, b, size",
    [
        (np.array(0.5, dtype=config.floatX), np.array(0.5, dtype=config.floatX), None),
        (np.array(0.5, dtype=config.floatX), np.array(0.5, dtype=config.floatX), []),
        (
            np.full((1, 2), 0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_gamma_samples(a, b, size):
    gamma_test_fn = fixed_scipy_rvs("gamma")

    def test_fn(shape, rate, **kwargs):
        return gamma_test_fn(shape, scale=1.0 / rate, **kwargs)

    compare_sample_values(
        gamma,
        a,
        b,
        size=size,
        test_fn=test_fn,
    )


@pytest.mark.parametrize(
    "df, size",
    [
        (np.array(2, dtype=config.floatX), None),
        (np.array(2, dtype=config.floatX), []),
        (np.full((1, 2), 2, dtype=np.int64), None),
    ],
)
def test_chisquare_samples(df, size):
    compare_sample_values(chisquare, df, size=size, test_fn=fixed_scipy_rvs("chi2"))


@pytest.mark.parametrize(
    "mu, beta, size",
    [
        (np.array(0, dtype=config.floatX), np.array(1, dtype=config.floatX), None),
        (np.array(0, dtype=config.floatX), np.array(1, dtype=config.floatX), []),
        (
            np.full((1, 2), 0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_gumbel_samples(mu, beta, size):
    compare_sample_values(
        gumbel, mu, beta, size=size, test_fn=fixed_scipy_rvs("gumbel_r")
    )


@pytest.mark.parametrize(
    "lam, size",
    [
        (np.array(10, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), []),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            None,
        ),
    ],
)
def test_exponential_samples(lam, size):
    compare_sample_values(exponential, lam, size=size)


def test_exponential_default_args():
    compare_sample_values(exponential)


@pytest.mark.parametrize(
    "alpha, size",
    [
        (np.array(10, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), []),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            None,
        ),
    ],
)
def test_weibull_samples(alpha, size):
    compare_sample_values(weibull, alpha, size=size)


@pytest.mark.parametrize(
    "loc, scale, size",
    [
        (np.array(2, dtype=config.floatX), np.array(0.5, dtype=config.floatX), None),
        (np.array(2, dtype=config.floatX), np.array(0.5, dtype=config.floatX), []),
        (
            np.full((1, 2), 2, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_logistic_samples(loc, scale, size):
    compare_sample_values(logistic, loc, scale, size=size)


def test_logistic_default_args():
    compare_sample_values(logistic)


@pytest.mark.parametrize(
    "mu, kappa, size",
    [
        (
            np.array(np.pi, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
        (np.array(np.pi, dtype=config.floatX), np.array(0.5, dtype=config.floatX), []),
        (
            np.full((1, 2), np.pi, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_vonmises_samples(mu, kappa, size):
    compare_sample_values(vonmises, mu, kappa, size=size)


@pytest.mark.parametrize(
    "alpha, size",
    [
        (np.array(0.5, dtype=config.floatX), None),
        (np.array(0.5, dtype=config.floatX), []),
        (
            np.full((1, 2), 0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_pareto_samples(alpha, size):
    compare_sample_values(pareto, alpha, size=size, test_fn=fixed_scipy_rvs("pareto"))


def mvnormal_test_fn(mean=None, cov=None, size=None, random_state=None):
    if mean is None:
        mean = np.array([0.0], dtype=config.floatX)
    if cov is None:
        cov = np.array([[1.0]], dtype=config.floatX)
    if size is None:
        size = ()
    return multivariate_normal.rng_fn(random_state, mean, cov, size)


@pytest.mark.parametrize(
    "mu, cov, size",
    [
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            None,
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [1],
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [4],
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [4, 1],
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [4, 1, 1],
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [1, 4, 1],
        ),
        (
            np.array([0], dtype=config.floatX),
            np.eye(1, dtype=config.floatX),
            [1, 5, 8],
        ),
        (
            np.array([0, 1, 2], dtype=config.floatX),
            np.diag(
                np.array([1, 10, 100], dtype=config.floatX),
            ),
            None,
        ),
        (
            np.array([0, 1, 2], dtype=config.floatX),
            np.stack(
                [
                    np.eye(3, dtype=config.floatX),
                    np.eye(3, dtype=config.floatX) * 10.0,
                ]
            ),
            [2, 3, 2],
        ),
        (
            np.array([[0, 1, 2], [4, 5, 6]], dtype=config.floatX),
            np.diag(
                np.array([1, 10, 100], dtype=config.floatX),
            ),
            None,
        ),
        (
            np.array([[0, 1, 2], [4, 5, 6]], dtype=config.floatX),
            np.stack(
                [
                    np.eye(3, dtype=config.floatX),
                    np.eye(3, dtype=config.floatX) * 10.0,
                ]
            ),
            [2, 3, 2, 2],
        ),
        (
            np.array([[0], [10], [100]], dtype=config.floatX),
            np.eye(1, dtype=config.floatX) * 1e-6,
            [2, 3, 3],
        ),
    ],
)
def test_mvnormal_samples(mu, cov, size):
    compare_sample_values(
        multivariate_normal, mu, cov, size=size, test_fn=mvnormal_test_fn
    )


def test_mvnormal_default_args():
    compare_sample_values(multivariate_normal, test_fn=mvnormal_test_fn)

    with pytest.raises(ValueError, match="operands could not be broadcast together "):
        multivariate_normal.rng_fn(
            None, np.zeros((3, 2)), np.ones((3, 2, 2)), size=(4,)
        )


@config.change_flags(compute_test_value="raise")
def test_mvnormal_ShapeFeature():
    M_at = iscalar("M")
    M_at.tag.test_value = 2

    d_rv = multivariate_normal(at.ones((M_at,)), at.eye(M_at), size=2)

    fg = FunctionGraph(
        [i for i in graph_inputs([d_rv]) if not isinstance(i, Constant)],
        [d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert get_test_value(s1) == 2
    assert M_at in graph_inputs([s2])

    # Test broadcasted shapes
    mean = tensor(config.floatX, shape=(1, None))
    mean.tag.test_value = np.array([[0, 1, 2]], dtype=config.floatX)

    test_covar = np.diag(np.array([1, 10, 100], dtype=config.floatX))
    test_covar = np.stack([test_covar, test_covar * 10.0])
    cov = at.as_tensor(test_covar).type()
    cov.tag.test_value = test_covar

    d_rv = multivariate_normal(mean, cov, size=[2, 3, 2])

    fg = FunctionGraph(
        outputs=[d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2, s3, s4 = fg.shape_feature.shape_of[d_rv]

    assert s1.get_test_value() == 2
    assert s2.get_test_value() == 3
    assert s3.get_test_value() == 2
    assert s4.get_test_value() == 3


@pytest.mark.parametrize(
    "alphas, size",
    [
        (np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX), None),
        (
            np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX),
            (10, 3),
        ),
        (
            np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX),
            (10, 2, 3),
        ),
    ],
)
def test_dirichlet_samples(alphas, size):
    def dirichlet_test_fn(mean=None, cov=None, size=None, random_state=None):
        if size is None:
            size = ()
        return dirichlet.rng_fn(random_state, alphas, size)

    compare_sample_values(dirichlet, alphas, size=size, test_fn=dirichlet_test_fn)


def test_dirichlet_rng():
    alphas = np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=config.floatX)

    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        # The independent dimension's shape cannot be broadcasted from (3,) to (10, 2)
        dirichlet.rng_fn(None, alphas, size=(10, 2))

    with pytest.raises(
        ValueError, match="input operand has more dimensions than allowed"
    ):
        # One of the independent dimension's shape is missing from size
        # (i.e. should be `(1, 3)`)
        dirichlet.rng_fn(None, np.broadcast_to(alphas, (1, 3, 3)), size=(3,))


M_at = iscalar("M")
M_at.tag.test_value = 3


@pytest.mark.parametrize(
    "M, size",
    [
        (at.ones((M_at,)), ()),
        (at.ones((M_at,)), (M_at + 1,)),
        (at.ones((M_at,)), (2, M_at)),
        (at.ones((M_at, M_at + 1)), ()),
        (at.ones((M_at, M_at + 1)), (M_at + 2, M_at)),
        (at.ones((M_at, M_at + 1)), (2, M_at + 2, M_at + 3, M_at)),
    ],
)
def test_dirichlet_infer_shape(M, size):
    rv = dirichlet(M, size=size)
    rv_shape = list(dirichlet._infer_shape(size or (), [M], None))

    all_args = (M,) + size
    fn_inputs = [
        i
        for i in graph_inputs([a for a in all_args if isinstance(a, Variable)])
        if not isinstance(i, (Constant, SharedVariable))
    ]
    aesara_fn = function(
        fn_inputs, [at.as_tensor(o) for o in rv_shape + [rv]], mode=py_mode
    )

    *rv_shape_val, rv_val = aesara_fn(
        *[
            i.tag.test_value
            for i in fn_inputs
            if not isinstance(i, (SharedVariable, Constant))
        ]
    )

    assert tuple(rv_shape_val) == tuple(rv_val.shape)


@config.change_flags(compute_test_value="raise")
def test_dirichlet_ShapeFeature():
    """Make sure `RandomVariable.infer_shape` works with `ShapeFeature`."""
    M_at = iscalar("M")
    M_at.tag.test_value = 2
    N_at = iscalar("N")
    N_at.tag.test_value = 3

    d_rv = dirichlet(at.ones((M_at, N_at)), name="Gamma")

    fg = FunctionGraph(
        outputs=[d_rv],
        clone=False,
        features=[ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[d_rv]

    assert M_at in graph_inputs([s1])
    assert N_at in graph_inputs([s2])


@pytest.mark.parametrize(
    "lam, size",
    [
        (np.array(10, dtype=np.int64), None),
        (np.array(10, dtype=np.int64), []),
        (
            np.full((1, 2), 10, dtype=np.int64),
            None,
        ),
    ],
)
def test_poisson_samples(lam, size):
    compare_sample_values(poisson, lam, size=size)


def test_poisson_default_args():
    compare_sample_values(poisson)


@pytest.mark.parametrize(
    "p, size",
    [
        (np.array(0.1, dtype=config.floatX), None),
        (np.array(0.1, dtype=config.floatX), []),
        (
            np.full((1, 2), 0.1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_geometric_samples(p, size):
    compare_sample_values(geometric, p, size=size)


@pytest.mark.parametrize(
    "ngood, nbad, nsample, size",
    [
        (
            np.array(10, dtype=np.int64),
            np.array(20, dtype=np.int64),
            np.array(5, dtype=np.int64),
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(20, dtype=np.int64),
            np.array(5, dtype=np.int64),
            [],
        ),
        (
            np.full((1, 2), 10, dtype=np.int64),
            np.array(20, dtype=np.int64),
            np.array(5, dtype=np.int64),
            None,
        ),
    ],
)
def test_hypergeometric_samples(ngood, nbad, nsample, size):
    compare_sample_values(hypergeometric, ngood, nbad, nsample, size=size)


@pytest.mark.parametrize(
    "loc, scale, size",
    [
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), None),
        (
            np.array([[0]], dtype=config.floatX),
            np.array([[1]], dtype=config.floatX),
            None,
        ),
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), []),
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), [2, 3]),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(0.1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_cauchy_samples(loc, scale, size):
    compare_sample_values(
        cauchy, loc, scale, size=size, test_fn=fixed_scipy_rvs("cauchy")
    )


def test_cauchy_default_args():
    compare_sample_values(cauchy, test_fn=stats.cauchy.rvs)


@pytest.mark.parametrize(
    "loc, scale, size",
    [
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), []),
        (np.array(10, dtype=config.floatX), np.array(0.1, dtype=config.floatX), [2, 3]),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(0.1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_halfcauchy_samples(loc, scale, size):
    compare_sample_values(
        halfcauchy, loc, scale, size=size, test_fn=fixed_scipy_rvs("halfcauchy")
    )


def test_halfcauchy_default_args():
    compare_sample_values(halfcauchy, test_fn=stats.halfcauchy.rvs)


@pytest.mark.parametrize(
    "loc, scale, size",
    [
        (np.array(2, dtype=config.floatX), np.array(1, dtype=config.floatX), None),
        (np.array(2, dtype=config.floatX), np.array(1, dtype=config.floatX), []),
        (np.array(2, dtype=config.floatX), np.array(1, dtype=config.floatX), [2, 3]),
        (
            np.full((1, 2), 2, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_invgamma_samples(loc, scale, size):
    compare_sample_values(
        invgamma,
        loc,
        scale,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: invgamma.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "mean, scale, size",
    [
        (np.array(10, dtype=config.floatX), np.array(1, dtype=config.floatX), None),
        (np.array(10, dtype=config.floatX), np.array(1, dtype=config.floatX), []),
        (np.array(10, dtype=config.floatX), np.array(1, dtype=config.floatX), [2, 3]),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_wald_samples(mean, scale, size):
    compare_sample_values(wald, mean, scale, size=size)


@pytest.mark.parametrize(
    "b, loc, scale, size",
    [
        (
            np.array(5, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
        (
            np.array(5, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            [],
        ),
        (
            np.array(5, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 5, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_truncexpon_samples(b, loc, scale, size):
    compare_sample_values(
        truncexpon,
        b,
        loc,
        scale,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: truncexpon.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "df, loc, scale, size",
    [
        (
            np.array(2, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
        (
            np.array(2, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            [],
        ),
        (
            np.array(2, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 5, dtype=config.floatX),
            np.array(0, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            None,
        ),
    ],
)
def test_t_samples(df, loc, scale, size):
    compare_sample_values(
        t,
        df,
        loc,
        scale,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: t.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "p, size",
    [
        (
            np.array(0.5, dtype=config.floatX),
            None,
        ),
        (
            np.array(0.5, dtype=config.floatX),
            [],
        ),
        (
            np.array(0.5, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_bernoulli_samples(p, size):
    compare_sample_values(
        bernoulli,
        p,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: bernoulli.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "loc, scale, size",
    [
        (
            np.array(10, dtype=config.floatX),
            np.array(5, dtype=config.floatX),
            None,
        ),
        (
            np.array(10, dtype=config.floatX),
            np.array(5, dtype=config.floatX),
            [],
        ),
        (
            np.array(10, dtype=config.floatX),
            np.array(5, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 10, dtype=config.floatX),
            np.array(5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_laplace_samples(loc, scale, size):
    compare_sample_values(laplace, loc, scale, size=size)


@pytest.mark.parametrize(
    "M, p, size",
    [
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            [],
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_binomial_samples(M, p, size):
    compare_sample_values(binomial, M, p, size=size)


@pytest.mark.parametrize(
    "M, p, size",
    [
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            [],
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_nbinom_samples(M, p, size):
    compare_sample_values(
        nbinom,
        M,
        p,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: nbinom.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "M, a, p, size",
    [
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            [],
        ),
        (
            np.array(10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 10, dtype=np.int64),
            np.array(0.5, dtype=config.floatX),
            np.array(0.5, dtype=config.floatX),
            None,
        ),
    ],
)
def test_betabinom_samples(M, a, p, size):
    compare_sample_values(
        betabinom,
        M,
        a,
        p,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: betabinom.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "alpha, p, lambd, size",
    [
        (
            np.array(2, dtype=config.floatX),
            np.array(3, dtype=config.floatX),
            np.array(5, dtype=config.floatX),
            None,
        ),
        (
            np.array(1, dtype=config.floatX),
            np.array(1, dtype=config.floatX),
            np.array(10, dtype=config.floatX),
            [],
        ),
        (
            np.array(2, dtype=config.floatX),
            np.array(2, dtype=config.floatX),
            np.array(10, dtype=config.floatX),
            [2, 3],
        ),
        (
            np.full((1, 2), 2, dtype=config.floatX),
            np.array(2, dtype=config.floatX),
            np.array(10, dtype=config.floatX),
            None,
        ),
    ],
)
def test_gengamma_samples(alpha, p, lambd, size):
    compare_sample_values(
        gengamma,
        alpha,
        p,
        lambd,
        size=size,
        test_fn=lambda *args, size=None, random_state=None, **kwargs: gengamma.rng_fn(
            random_state, *(args + (size,))
        ),
    )


@pytest.mark.parametrize(
    "M, p, size, test_fn",
    [
        (
            np.array(10, dtype=np.int64),
            np.array([0.7, 0.3], dtype=config.floatX),
            None,
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array([0.7, 0.3], dtype=config.floatX),
            [],
            None,
        ),
        (
            np.array(10, dtype=np.int64),
            np.array([0.7, 0.3], dtype=config.floatX),
            [2, 3],
            None,
        ),
        (
            np.full((1, 2), 10, dtype=np.int64),
            np.array([0.7, 0.3], dtype=config.floatX),
            None,
            lambda *args, size=None, random_state=None, **kwargs: multinomial.rng_fn(
                random_state, *(args + (size,))
            ),
        ),
        (
            np.array([10, 20], dtype=np.int64),
            np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX),
            None,
            lambda *args, **kwargs: np.array([[10, 0], [0, 20]]),
        ),
        (
            np.array([10, 20], dtype=np.int64),
            np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX),
            (3, 2),
            lambda *args, **kwargs: np.stack([np.array([[10, 0], [0, 20]])] * 3),
        ),
    ],
)
def test_multinomial_samples(M, p, size, test_fn):
    rng = np.random.default_rng(1234)
    compare_sample_values(
        multinomial,
        M,
        p,
        size=size,
        test_fn=test_fn,
        rng=rng,
    )


def test_multinomial_rng():
    test_M = np.array([10, 20], dtype=np.int64)
    test_p = np.array([[0.999, 0.001], [0.001, 0.999]], dtype=config.floatX)

    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        # The independent dimension's shape cannot be broadcasted from (2,) to (1,)
        multinomial.rng_fn(None, test_M, test_p, size=(1,))

    with pytest.raises(
        ValueError, match="input operand has more dimensions than allowed"
    ):
        # One of the independent dimension's shape is missing from size
        # (i.e. should be `(5, 2)`)
        multinomial.rng_fn(None, np.broadcast_to(test_M, (5, 2)), test_p, size=(2,))


@pytest.mark.parametrize(
    "p, size, test_fn",
    [
        (
            np.array([100000, 1, 1], dtype=config.floatX),
            None,
            lambda *args, **kwargs: np.array(0, dtype=np.int64),
        ),
        (
            np.array(
                [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]], dtype=config.floatX
            ),
            (10, 3),
            lambda *args, **kwargs: np.tile(np.arange(3).astype(np.int64), (10, 1)),
        ),
        (
            np.array(
                [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]], dtype=config.floatX
            ),
            (10, 2, 3),
            lambda *args, **kwargs: np.tile(np.arange(3).astype(np.int64), (10, 2, 1)),
        ),
        (
            np.full((4, 1, 3), [100000, 1, 1], dtype=config.floatX),
            (4, 2),
            lambda *args, **kwargs: np.zeros((4, 2), dtype=np.int64),
        ),
    ],
)
def test_categorical_samples(p, size, test_fn):
    p = p / p.sum(axis=-1, keepdims=True)
    rng = np.random.default_rng(232)

    compare_sample_values(
        categorical,
        p,
        size=size,
        test_fn=test_fn,
        rng=rng,
    )


def test_categorical_basic():
    p = np.array([[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]], dtype=config.floatX)
    p = p / p.sum(axis=-1)

    rng = np.random.default_rng()

    with pytest.raises(ValueError):
        # The independent dimension of p has shape=(3,) which cannot be
        # broadcasted to (10,)
        categorical.rng_fn(rng, p, size=(10,))

    msg = re.escape("`size` is incompatible with the shape of `p`")
    with pytest.raises(ValueError, match=msg):
        # The independent dimension of p has shape=(3,) which cannot be
        # broadcasted to (1,)
        categorical.rng_fn(rng, p, size=(1,))

    with pytest.raises(ValueError, match=msg):
        # The independent dimensions of p have shape=(1, 3) which cannot be
        # broadcasted to (3,)
        categorical.rng_fn(rng, p[None], size=(3,))


def test_randint_samples():

    with pytest.raises(TypeError):
        randint(10, rng=shared(np.random.default_rng()))

    rng = np.random.RandomState(2313)
    compare_sample_values(randint, 10, None, rng=rng)
    compare_sample_values(randint, 0, 1, rng=rng)
    compare_sample_values(randint, 0, 1, size=[3], rng=rng)
    compare_sample_values(randint, [0, 1, 2], 5, rng=rng)
    compare_sample_values(randint, [0, 1, 2], 5, size=[3, 3], rng=rng)
    compare_sample_values(randint, [0], [5], size=[1], rng=rng)
    compare_sample_values(randint, at.as_tensor_variable([-1]), [1], size=[1], rng=rng)
    compare_sample_values(
        randint,
        at.as_tensor_variable([-1]),
        [1],
        size=at.as_tensor_variable([1]),
        rng=rng,
    )


def test_integers_samples():

    with pytest.raises(TypeError):
        integers(10, rng=shared(np.random.RandomState()))

    rng = np.random.default_rng(2313)
    compare_sample_values(integers, 10, None, rng=rng)
    compare_sample_values(integers, 0, 1, rng=rng)
    compare_sample_values(integers, 0, 1, size=[3], rng=rng)
    compare_sample_values(integers, [0, 1, 2], 5, rng=rng)
    compare_sample_values(integers, [0, 1, 2], 5, size=[3, 3], rng=rng)
    compare_sample_values(integers, [0], [5], size=[1], rng=rng)
    compare_sample_values(integers, at.as_tensor_variable([-1]), [1], size=[1], rng=rng)
    compare_sample_values(
        integers,
        at.as_tensor_variable([-1]),
        [1],
        size=at.as_tensor_variable([1]),
        rng=rng,
    )


def test_choice_samples():
    with pytest.raises(NotImplementedError):
        choice._supp_shape_from_params(np.asarray(5))

    compare_sample_values(choice, np.asarray(5))
    compare_sample_values(choice, np.asarray([5]))
    compare_sample_values(choice, np.array([1.0, 5.0], dtype=config.floatX))
    compare_sample_values(choice, np.asarray([5]), 3)

    compare_sample_values(choice, np.array([[1, 2], [3, 4]]))
    compare_sample_values(choice, np.array([[1, 2], [3, 4]]), p=[0.4, 0.6])

    compare_sample_values(choice, [1, 2, 3], 1)

    compare_sample_values(
        choice, [1, 2, 3], 1, p=at.as_tensor([1 / 3.0, 1 / 3.0, 1 / 3.0])
    )

    # p must be 1-dimensional.
    # TODO: The exception is raised at runtime but could be raised at compile
    # time in some situations using static shape analysis.
    with pytest.raises(ValueError):
        rng = np.random.default_rng()
        rng_at = shared(rng, borrow=True)
        choice(a=[1, 2], p=at.as_tensor([[0.1, 0.9], [0.3, 0.7]]), rng=rng_at).eval()

    compare_sample_values(choice, [1, 2, 3], (10, 2), replace=True)
    compare_sample_values(choice, at.as_tensor_variable([1, 2, 3]), 2, replace=True)


def test_permutation_samples():
    compare_sample_values(
        permutation,
        np.asarray(5),
        test_fn=lambda x, random_state=None: random_state.permutation(x.item()),
    )
    compare_sample_values(permutation, [1, 2, 3])
    compare_sample_values(permutation, [[1, 2], [3, 4]])
    compare_sample_values(permutation, np.array([1.0, 2.0, 3.0], dtype=config.floatX))


@config.change_flags(compute_test_value="off")
def test_pickle():
    # This is an interesting `Op` case, because it has `None` types and a
    # conditional dtype
    sample_a = choice(5, size=(2, 3))

    a_pkl = pickle.dumps(sample_a)
    a_unpkl = pickle.loads(a_pkl)

    assert a_unpkl.owner.op._props() == sample_a.owner.op._props()

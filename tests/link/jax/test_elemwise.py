import numpy as np
import pytest
import scipy.special

import aesara
import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.tensor import elemwise as at_elemwise
from aesara.tensor.math import all as at_all
from aesara.tensor.math import prod
from aesara.tensor.math import sum as at_sum
from aesara.tensor.special import SoftmaxGrad, log_softmax, softmax
from aesara.tensor.type import matrix, tensor, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_Dimshuffle():
    a_at = matrix("a")

    x = a_at.T
    x_fg = FunctionGraph([a_at], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    x = a_at.dimshuffle([0, 1, "x"])
    x_fg = FunctionGraph([a_at], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    a_at = tensor(dtype=config.floatX, shape=(None, 1))
    x = a_at.dimshuffle((0,))
    x_fg = FunctionGraph([a_at], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])

    a_at = tensor(dtype=config.floatX, shape=(None, 1))
    x = at_elemwise.DimShuffle([False, True], (0,))(a_at)
    x_fg = FunctionGraph([a_at], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])


def test_jax_CAReduce():
    a_at = vector("a")
    a_at.tag.test_value = np.r_[1, 2, 3].astype(config.floatX)

    x = at_sum(a_at, axis=None)
    x_fg = FunctionGraph([a_at], [x])

    compare_jax_and_py(x_fg, [np.r_[1, 2, 3].astype(config.floatX)])

    a_at = matrix("a")
    a_at.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = at_sum(a_at, axis=0)
    x_fg = FunctionGraph([a_at], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = at_sum(a_at, axis=1)
    x_fg = FunctionGraph([a_at], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    a_at = matrix("a")
    a_at.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = prod(a_at, axis=0)
    x_fg = FunctionGraph([a_at], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = at_all(a_at)
    x_fg = FunctionGraph([a_at], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax(axis):
    x = matrix("x")
    x.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_logsoftmax(axis):
    x = matrix("x")
    x.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = log_softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax_grad(axis):
    dy = matrix("dy")
    dy.tag.test_value = np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
    sm = matrix("sm")
    sm.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = SoftmaxGrad(axis=axis)(dy, sm)
    fgraph = FunctionGraph([dy, sm], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000), (10000, 10000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark(size, axis, benchmark):
    X = at.matrix("X")
    X_max = at.max(X, axis=axis, keepdims=True)
    X_max = at.switch(at.isinf(X_max), 0, X_max)
    X_lse = at.log(at.sum(at.exp(X - X_max), axis=axis, keepdims=True)) + X_max

    X_val = np.random.normal(size=size)

    X_lse_fn = aesara.function([X], X_lse, mode="JAX")

    # JIT compile first
    _ = X_lse_fn(X_val)

    res = benchmark(X_lse_fn, X_val)

    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)

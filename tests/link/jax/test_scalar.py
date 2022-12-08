import numpy as np
import pytest

import aesara.scalar.basic as aes
import aesara.tensor as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.scalar.basic import Composite
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.math import all as at_all
from aesara.tensor.math import (
    cosh,
    erf,
    erfc,
    erfinv,
    log,
    log1mexp,
    psi,
    sigmoid,
    softplus,
)
from aesara.tensor.type import matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_second():
    a0 = scalar("a0")
    b = scalar("b")

    out = aes.second(a0, b)
    fgraph = FunctionGraph([a0, b], [out])
    compare_jax_and_py(fgraph, [10.0, 5.0])

    a1 = vector("a1")
    out = at.second(a1, b)
    fgraph = FunctionGraph([a1, b], [out])
    compare_jax_and_py(fgraph, [np.zeros([5], dtype=config.floatX), 5.0])


def test_identity():
    a = scalar("a")
    a.tag.test_value = 10

    out = aes.identity(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize(
    "x, y, x_val, y_val",
    [
        (scalar("x"), scalar("y"), np.array(10), np.array(20)),
        (scalar("x"), vector("y"), np.array(10), np.arange(10, 20)),
        (
            matrix("x"),
            vector("y"),
            np.arange(10 * 20).reshape((20, 10)),
            np.arange(10, 20),
        ),
    ],
)
def test_jax_Composite(x, y, x_val, y_val):
    x_s = aes.float64("x")
    y_s = aes.float64("y")

    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + aes.exp(x_s - y_s)]))

    out = comp_op(x, y)

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        x_val.astype(config.floatX),
        y_val.astype(config.floatX),
    ]
    _ = compare_jax_and_py(out_fg, test_input_vals)


def test_erf():
    x = scalar("x")
    out = erf(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [1.0])


def test_erfc():
    x = scalar("x")
    out = erfc(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [1.0])


def test_erfinv():
    x = scalar("x")
    out = erfinv(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [1.0])


def test_psi():
    x = scalar("x")
    out = psi(x)
    fg = FunctionGraph([x], [out])
    compare_jax_and_py(fg, [3.0])


def test_log1mexp():
    x = vector("x")
    out = log1mexp(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [[-1.0, -0.75, -0.5, -0.25]])


def test_nnet():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    out = sigmoid(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = softplus(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_variadic_Scalar():
    mu = vector("mu", dtype=config.floatX)
    mu.tag.test_value = np.r_[0.1, 1.1].astype(config.floatX)
    tau = vector("tau", dtype=config.floatX)
    tau.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    res = -tau * mu

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    res = -tau * (tau - mu) ** 2

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_add_scalars():
    x = at.matrix("x")
    size = x.shape[0] + x.shape[0] + x.shape[1]
    out = at.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((2, 3)).astype(config.floatX)])


def test_mul_scalars():
    x = at.matrix("x")
    size = x.shape[0] * x.shape[0] * x.shape[1]
    out = at.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((2, 3)).astype(config.floatX)])


def test_div_scalars():
    x = at.matrix("x")
    size = x.shape[0] // x.shape[1]
    out = at.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((12, 3)).astype(config.floatX)])


def test_mod_scalars():
    x = at.matrix("x")
    size = x.shape[0] % x.shape[1]
    out = at.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((12, 3)).astype(config.floatX)])


def test_jax_multioutput():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    y = vector("y")
    y.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)

    w = cosh(x**2 + y / 3.0)
    v = cosh(x / 3.0 + y**2)

    fgraph = FunctionGraph([x, y], [w, v])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_logp():
    mu = vector("mu")
    mu.tag.test_value = np.r_[0.0, 0.0].astype(config.floatX)
    tau = vector("tau")
    tau.tag.test_value = np.r_[1.0, 1.0].astype(config.floatX)
    sigma = vector("sigma")
    sigma.tag.test_value = (1.0 / get_test_value(tau)).astype(config.floatX)
    value = vector("value")
    value.tag.test_value = np.r_[0.1, -10].astype(config.floatX)

    logp = (-tau * (value - mu) ** 2 + log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = at_all([at_all(1 * val) for val in conditions])
    normal_logp = at.switch(alltrue, logp, -np.inf)

    fgraph = FunctionGraph([mu, tau, sigma, value], [normal_logp])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

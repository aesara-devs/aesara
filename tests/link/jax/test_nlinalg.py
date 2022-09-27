import numpy as np
import pytest
from packaging.version import parse as version_parse

from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.graph.rewriting.db import RewriteDatabaseQuery
from aesara.link.jax import JAXLinker
from aesara.tensor import blas as at_blas
from aesara.tensor import nlinalg as at_nlinalg
from aesara.tensor.math import MaxAndArgmax
from aesara.tensor.math import max as at_max
from aesara.tensor.math import maximum
from aesara.tensor.type import dvector, matrix, scalar, tensor3, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a.tag.test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b.tag.test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = at_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [get_test_value(a)[:-1], get_test_value(b)]
    opts = RewriteDatabaseQuery(include=[None], exclude=["cxx_only", "BlasOpt"])
    jax_mode = Mode(JAXLinker(), opts)
    aesara_jax_fn = function(fgraph.inputs, fgraph.outputs, mode=jax_mode)
    with pytest.raises(TypeError):
        aesara_jax_fn(*inputs)

    # matrix . matrix
    a = matrix("a")
    a.tag.test_value = np.linspace(-1, 1, 5 * 3).astype(config.floatX).reshape((5, 3))
    b = matrix("b")
    b.tag.test_value = np.linspace(1, -1, 5 * 3).astype(config.floatX).reshape((5, 3))
    out = at_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_basic_multiout():
    rng = np.random.default_rng(213234)

    M = rng.normal(size=(3, 3))
    X = M.dot(M.T)

    x = matrix("x")

    outs = at_nlinalg.eig(x)
    out_fg = FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(config.floatX), y, rtol=1e-3)

    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = at_nlinalg.eigh(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = at_nlinalg.qr(x, mode="full")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = at_nlinalg.qr(x, mode="reduced")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = at_nlinalg.svd(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_basic_multiout_omni():
    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = dvector()
    mx, amx = MaxAndArgmax([0])(x)
    out = mx * amx
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.r_[1, 2]])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_tensor_basics():
    y = vector("y")
    y.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A.tag.test_value = np.empty((2, 2), dtype=config.floatX)
    alpha = scalar("alpha")
    alpha.tag.test_value = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta.tag.test_value = np.array(5.0, dtype=config.floatX)

    # This should be converted into a `Gemv` `Op` when the non-JAX compatible
    # optimizations are turned on; however, when using JAX mode, it should
    # leave the expression alone.
    out = y.dot(alpha * A).dot(x) + beta * y
    fgraph = FunctionGraph([y, x, A, alpha, beta], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = maximum(y, x)
    fgraph = FunctionGraph([y, x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = at_max(y)
    fgraph = FunctionGraph([y], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

import numpy as np
import pytest
from packaging.version import parse as version_parse

import aesara.tensor.basic as at
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import get_test_value
from aesara.tensor import extra_ops as at_extra_ops
from aesara.tensor.type import matrix, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def test_extra_ops():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = at_extra_ops.cumsum(a, axis=0)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = at_extra_ops.cumprod(a, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = at_extra_ops.diff(a, n=2, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = at_extra_ops.repeat(a, (3, 3), axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    c = at.as_tensor(5)

    out = at_extra_ops.fill_diagonal(a, c)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = at_extra_ops.fill_diagonal_offset(a, c, c)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = at_extra_ops.Unique(axis=1)(a)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    indices = np.arange(np.prod((3, 4)))
    out = at_extra_ops.unravel_index(indices, (3, 4), order="C")
    fgraph = FunctionGraph([], out)
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )


@pytest.mark.parametrize(
    "x, shape",
    [
        (
            set_test_value(
                vector("x"), np.random.random(size=(2,)).astype(config.floatX)
            ),
            [at.as_tensor(3, dtype=np.int64), at.as_tensor(2, dtype=np.int64)],
        ),
        (
            set_test_value(
                vector("x"), np.random.random(size=(2,)).astype(config.floatX)
            ),
            [at.as_tensor(3, dtype=np.int8), at.as_tensor(2, dtype=np.int64)],
        ),
    ],
)
def test_BroadcastTo(x, shape):
    out = at_extra_ops.broadcast_to(x, shape)
    fgraph = FunctionGraph(outputs=[out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_extra_ops_omni():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    # This function also cannot take symbolic input.
    c = at.as_tensor(5)
    out = at_extra_ops.bartlett(c)
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    multi_index = np.unravel_index(np.arange(np.prod((3, 4))), (3, 4))
    out = at_extra_ops.ravel_multi_index(multi_index, (3, 4))
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )

    # The inputs are "concrete", yet it still has problems?
    out = at_extra_ops.Unique()(
        at.as_tensor(np.arange(6, dtype=config.floatX).reshape((3, 2)))
    )
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_unique_nonconcrete():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = at_extra_ops.Unique()(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

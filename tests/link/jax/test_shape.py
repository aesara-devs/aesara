import numpy as np
import pytest

import aesara.tensor as at
from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor.shape import Shape, Shape_i, Unbroadcast, reshape
from aesara.tensor.type import iscalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_shape_ops():
    x_np = np.zeros((20, 3))
    x = Shape()(at.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)

    x = Shape_i(1)(at.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)


def test_jax_specify_shape():
    in_at = at.matrix("in")
    x = at.specify_shape(in_at, (4, 5))
    x_fg = FunctionGraph([in_at], [x])
    compare_jax_and_py(x_fg, [np.ones((4, 5)).astype(config.floatX)])

    # When used to assert two arrays have similar shapes
    in_at = at.matrix("in")
    shape_at = at.matrix("shape")
    x = at.specify_shape(in_at, shape_at.shape)
    x_fg = FunctionGraph([in_at, shape_at], [x])
    compare_jax_and_py(
        x_fg,
        [np.ones((4, 5)).astype(config.floatX), np.ones((4, 5)).astype(config.floatX)],
    )


def test_jax_Reshape_constant():
    a = vector("a")
    x = reshape(a, (2, 2))
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


def test_jax_Reshape_concrete_shape():
    """JAX should compile when a concrete value is passed for the `shape` parameter."""
    a = vector("a")
    x = reshape(a, a.shape)
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    x = reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


@pytest.mark.xfail(
    reason="`shape_at` should be specified as a static argument", strict=True
)
def test_jax_Reshape_shape_graph_input():
    a = vector("a")
    shape_at = iscalar("b")
    x = reshape(a, (shape_at, shape_at))
    x_fg = FunctionGraph([a, shape_at], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX), 2])


def test_jax_compile_ops():

    x = DeepCopyOp()(at.as_tensor_variable(1.1))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 1, 1))
    x = Unbroadcast(0, 2)(at.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x = ViewOp()(at.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

import jax
import numpy as np
import pytest
from packaging.version import parse as version_parse

import aesara.tensor as at
from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.tensor.shape import Shape, Shape_i, SpecifyShape, Unbroadcast, reshape
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


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_specify_shape():
    x_np = np.zeros((20, 3))
    x = SpecifyShape()(at.as_tensor_variable(x_np), (20, 3))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    with config.change_flags(compute_test_value="off"):

        x = SpecifyShape()(at.as_tensor_variable(x_np), *(2, 3))
        x_fg = FunctionGraph([], [x])

        with pytest.raises(AssertionError):
            compare_jax_and_py(x_fg, [])


def test_jax_Reshape():
    a = vector("a")
    x = reshape(a, (2, 2))
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    # Test breaking "omnistaging" changes in JAX.
    # See https://github.com/tensorflow/probability/commit/782d0c64eb774b9aac54a1c8488e4f1f96fbbc68
    x = reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    x_fg = FunctionGraph([a], [x])
    with pytest.raises(
        TypeError,
        match="Shapes must be 1D sequences of concrete values of integer type",
    ):
        compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    b = iscalar("b")
    x = reshape(a, (b, b))
    x_fg = FunctionGraph([a, b], [x])
    with pytest.raises(
        TypeError,
        match="Shapes must be 1D sequences of concrete values of integer type",
    ):
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

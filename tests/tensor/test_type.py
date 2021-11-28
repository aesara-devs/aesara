import os.path as path
from tempfile import mkdtemp

import numpy as np
import pytest

from aesara.configdefaults import config
from aesara.tensor.type import TensorType


def test_numpy_dtype():
    test_type = TensorType(np.int32, [])
    assert test_type.dtype == "int32"


def test_filter_variable():
    test_type = TensorType(config.floatX, [])

    with pytest.raises(TypeError):
        test_type.filter(test_type())

    test_type = TensorType(config.floatX, [True, False])

    with pytest.raises(TypeError):
        test_type.filter(np.empty((0, 1), dtype=config.floatX))

    with pytest.raises(TypeError, match=".*not aligned.*"):
        test_val = np.empty((1, 2), dtype=config.floatX)
        test_val.flags.aligned = False
        test_type.filter(test_val)

    with pytest.raises(ValueError, match="Non-finite"):
        test_type.filter_checks_isfinite = True
        test_type.filter(np.full((1, 2), np.inf, dtype=config.floatX))


def test_filter_strict():
    test_type = TensorType(config.floatX, [])

    with pytest.raises(TypeError):
        test_type.filter(1, strict=True)

    with pytest.raises(TypeError):
        test_type.filter(np.array(1, dtype=int), strict=True)


def test_filter_ndarray_subclass():
    """Make sure `TensorType.filter` can handle NumPy `ndarray` subclasses."""
    test_type = TensorType(config.floatX, [False])

    class MyNdarray(np.ndarray):
        pass

    test_val = np.array([1.0], dtype=config.floatX).view(MyNdarray)
    assert isinstance(test_val, MyNdarray)

    res = test_type.filter(test_val)
    assert isinstance(res, MyNdarray)
    assert res is test_val


def test_filter_float_subclass():
    """Make sure `TensorType.filter` can handle `float` subclasses."""
    with config.change_flags(floatX="float64"):
        test_type = TensorType("float64", broadcastable=[])

        nan = np.array([np.nan], dtype="float64")[0]
        assert isinstance(nan, float) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)

    with config.change_flags(floatX="float32"):
        # Try again, except this time `nan` isn't a `float`
        test_type = TensorType("float32", broadcastable=[])

        nan = np.array([np.nan], dtype="float32")[0]
        assert isinstance(nan, np.floating) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)


def test_filter_memmap():
    r"""Make sure `TensorType.filter` can handle NumPy `memmap`\s subclasses."""
    data = np.arange(12, dtype=config.floatX)
    data.resize((3, 4))
    filename = path.join(mkdtemp(), "newfile.dat")
    fp = np.memmap(filename, dtype=config.floatX, mode="w+", shape=(3, 4))

    test_type = TensorType(config.floatX, [False, False])

    res = test_type.filter(fp)
    assert res is fp


def test_tensor_values_eq_approx():
    # test, inf, -inf and nan equal themself
    a = np.asarray([-np.inf, -1, 0, 1, np.inf, np.nan])
    assert TensorType.values_eq_approx(a, a)

    # test inf, -inf don't equal themself
    b = np.asarray([np.inf, -1, 0, 1, np.inf, np.nan])
    assert not TensorType.values_eq_approx(a, b)
    b = np.asarray([-np.inf, -1, 0, 1, -np.inf, np.nan])
    assert not TensorType.values_eq_approx(a, b)

    # test allow_remove_inf
    b = np.asarray([np.inf, -1, 0, 1, 5, np.nan])
    assert TensorType.values_eq_approx(a, b, allow_remove_inf=True)
    b = np.asarray([np.inf, -1, 0, 1, 5, 6])
    assert not TensorType.values_eq_approx(a, b, allow_remove_inf=True)

    # test allow_remove_nan
    b = np.asarray([np.inf, -1, 0, 1, 5, np.nan])
    assert not TensorType.values_eq_approx(a, b, allow_remove_nan=False)
    b = np.asarray([-np.inf, -1, 0, 1, np.inf, 6])
    assert not TensorType.values_eq_approx(a, b, allow_remove_nan=False)

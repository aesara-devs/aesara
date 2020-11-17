import os.path as path
from tempfile import mkdtemp

import numpy as np
import pytest

from theano import change_flags, config
from theano.tensor.type import TensorType


def test_filter_variable():
    test_type = TensorType(config.floatX, [])

    with pytest.raises(TypeError):
        test_type.filter(test_type())


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
    with change_flags(floatX="float64"):
        test_type = TensorType("float64", broadcastable=[])

        nan = np.array([np.nan], dtype="float64")[0]
        assert isinstance(nan, np.float) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)

    with change_flags(floatX="float32"):
        # Try again, except this time `nan` isn't a `float`
        test_type = TensorType("float32", broadcastable=[])

        nan = np.array([np.nan], dtype="float32")[0]
        assert isinstance(nan, np.floating) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan)
        assert isinstance(filtered_nan, np.ndarray)


def test_filter_memmap():
    """Make sure `TensorType.filter` can handle NumPy `memmap`s subclasses."""
    data = np.arange(12, dtype=config.floatX)
    data.resize((3, 4))
    filename = path.join(mkdtemp(), "newfile.dat")
    fp = np.memmap(filename, dtype=config.floatX, mode="w+", shape=(3, 4))

    test_type = TensorType(config.floatX, [False, False])

    res = test_type.filter(fp)
    assert res is fp

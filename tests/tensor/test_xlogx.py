import numpy as np

import aesara
from aesara.tensor import as_tensor_variable
from aesara.tensor.xlogx import xlogx, xlogy0
from tests import unittest_tools as utt


def test_xlogx():
    x = as_tensor_variable([1, 0])
    y = xlogx(x)
    f = aesara.function([], y)
    assert np.array_equal(f(), np.asarray([0, 0.0]))

    rng = np.random.default_rng(24982)
    utt.verify_grad(xlogx, [rng.random((3, 4))])


def test_xlogy0():
    x = as_tensor_variable([1, 0])
    y = as_tensor_variable([1, 0])
    z = xlogy0(x, y)
    f = aesara.function([], z)
    assert np.array_equal(f(), np.asarray([0, 0.0]))

    rng = np.random.default_rng(24982)
    utt.verify_grad(xlogy0, [rng.random((3, 4)), rng.random((3, 4))])

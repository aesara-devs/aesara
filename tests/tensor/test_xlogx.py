import numpy.random

import aesara
from aesara.tensor import as_tensor_variable
from aesara.tensor.xlogx import xlogx, xlogy0
from tests import unittest_tools as utt


class TestXlogX:
    def test_basic(self):
        x = as_tensor_variable([1, 0])
        y = xlogx(x)
        f = aesara.function([], [y])
        assert numpy.all(f() == numpy.asarray([0, 0.0]))

        # class Dummy(object):
        #     def make_node(self, a):
        #         return [xlogx(a)[:,2]]
        utt.verify_grad(xlogx, [numpy.random.rand(3, 4)])


class TestXlogY0:
    def test_basic(self):
        utt.verify_grad(xlogy0, [numpy.random.rand(3, 4), numpy.random.rand(3, 4)])

        x = as_tensor_variable([1, 0])
        y = as_tensor_variable([1, 0])
        z = xlogy0(x, y)
        f = aesara.function([], z)
        assert numpy.all(f() == numpy.asarray([0, 0.0]))

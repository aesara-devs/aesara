import numpy.random

import theano
from tests import unittest_tools as utt
from theano.tensor import as_tensor_variable
from theano.tensor.xlogx import xlogx, xlogy0


class TestXlogX:
    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):
        x = as_tensor_variable([1, 0])
        y = xlogx(x)
        f = theano.function([], [y])
        assert numpy.all(f() == numpy.asarray([0, 0.0]))

        # class Dummy(object):
        #     def make_node(self, a):
        #         return [xlogx(a)[:,2]]
        utt.verify_grad(xlogx, [numpy.random.rand(3, 4)])


class TestXlogY0:
    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):
        utt.verify_grad(xlogy0, [numpy.random.rand(3, 4), numpy.random.rand(3, 4)])

        x = as_tensor_variable([1, 0])
        y = as_tensor_variable([1, 0])
        z = xlogy0(x, y)
        f = theano.function([], z)
        assert numpy.all(f() == numpy.asarray([0, 0.0]))

import numpy as np
import pytest

import theano
from tests.tensor.test_blas import TestBlasStrides, gemm_no_inplace
from tests.unittest_tools import OptimizationTestMixin
from theano import tensor as tt
from theano.tensor.blas_scipy import ScipyGer
from theano.tensor.math import outer
from theano.tensor.type import tensor


@pytest.mark.skipif(not theano.tensor.blas_scipy.have_fblas, reason="fblas needed")
class TestScipyGer(OptimizationTestMixin):
    def setup_method(self):
        self.mode = theano.compile.get_default_mode()
        self.mode = self.mode.including("fast_run")
        self.mode = self.mode.excluding("c_blas")  # c_blas trumps scipy Ops
        dtype = self.dtype = "float64"  # optimization isn't dtype-dependent
        self.A = tensor(dtype=dtype, broadcastable=(False, False))
        self.a = tensor(dtype=dtype, broadcastable=())
        self.x = tensor(dtype=dtype, broadcastable=(False,))
        self.y = tensor(dtype=dtype, broadcastable=(False,))
        self.Aval = np.ones((2, 3), dtype=dtype)
        self.xval = np.asarray([1, 2], dtype=dtype)
        self.yval = np.asarray([1.5, 2.7, 3.9], dtype=dtype)

    def function(self, inputs, outputs):
        return theano.function(inputs, outputs, self.mode)

    def run_f(self, f):
        f(self.Aval, self.xval, self.yval)
        f(self.Aval[::-1, ::-1], self.xval[::-1], self.yval[::-1])

    def b(self, bval):
        return tt.as_tensor_variable(np.asarray(bval, dtype=self.dtype))

    def test_outer(self):
        f = self.function([self.x, self.y], outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=True))

    def test_A_plus_outer(self):
        f = self.function([self.A, self.x, self.y], self.A + outer(self.x, self.y))
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness

    def test_A_plus_scaled_outer(self):
        f = self.function(
            [self.A, self.x, self.y], self.A + 0.1 * outer(self.x, self.y)
        )
        self.assertFunctionContains(f, ScipyGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness

    def test_scaled_A_plus_scaled_outer(self):
        f = self.function(
            [self.A, self.x, self.y], 0.2 * self.A + 0.1 * outer(self.x, self.y)
        )
        self.assertFunctionContains(f, gemm_no_inplace)
        self.run_f(f)  # DebugMode tests correctness


class TestBlasStridesScipy(TestBlasStrides):
    mode = theano.compile.get_default_mode()
    mode = mode.including("fast_run").excluding("gpu", "c_blas")

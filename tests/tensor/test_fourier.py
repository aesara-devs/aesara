import numpy as np
import pytest

import aesara
from aesara.tensor.fourier import Fourier, fft
from aesara.tensor.type import dmatrix, dvector, iscalar
from tests import unittest_tools as utt


class TestFourier(utt.InferShapeTester):
    rng = np.random.default_rng(43)

    def setup_method(self):
        super().setup_method()
        self.op_class = Fourier
        self.op = fft

    def test_perform(self):
        a = dmatrix()
        f = aesara.function([a], self.op(a, n=10, axis=0))
        a = np.random.random((8, 6))
        assert np.allclose(f(a), np.fft.fft(a, 10, 0))

    def test_infer_shape(self):
        a = dvector()
        self._compile_and_check(
            [a], [self.op(a, 16, 0)], [np.random.random(12)], self.op_class
        )
        a = dmatrix()
        for var in [
            self.op(a, 16, 1),
            self.op(a, None, 1),
            self.op(a, 16, None),
            self.op(a, None, None),
        ]:
            self._compile_and_check(
                [a], [var], [np.random.random((12, 4))], self.op_class
            )
        b = iscalar()
        for var in [self.op(a, 16, b), self.op(a, None, b)]:
            self._compile_and_check(
                [a, b], [var], [np.random.rand(12, 4), 0], self.op_class
            )

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_gradient(self):
        def fft_test1(a):
            return self.op(a, None, None)

        def fft_test2(a):
            return self.op(a, None, 0)

        def fft_test3(a):
            return self.op(a, 4, None)

        def fft_test4(a):
            return self.op(a, 4, 0)

        pts = [
            np.random.random((5, 2, 4, 3)),
            np.random.random((2, 3, 4)),
            np.random.random((2, 5)),
            np.random.random(5),
        ]
        for fft_test in [fft_test1, fft_test2, fft_test3, fft_test4]:
            for pt in pts:
                aesara.gradient.verify_grad(
                    fft_test, [pt], n_tests=1, rng=TestFourier.rng, out_type="complex64"
                )


if __name__ == "__main__":
    t = TestFourier("setUp")
    t.setup_method()
    t.test_perform()
    t.test_infer_shape()
    t.test_gradient()

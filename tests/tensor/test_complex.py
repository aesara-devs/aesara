import numpy as np
import pytest

import aesara
from aesara.gradient import GradientError
from aesara.tensor.basic import cast
from aesara.tensor.math import complex as at_complex
from aesara.tensor.math import complex_from_polar, imag, real
from aesara.tensor.type import cvector, dvector, fmatrix, fvector, imatrix, zvector
from tests import unittest_tools as utt


class TestRealImag:
    def test_basic(self):
        x = zvector()
        rng = np.random.default_rng(23)
        xval = np.asarray(
            list(
                complex(rng.standard_normal(), rng.standard_normal()) for i in range(10)
            )
        )
        assert np.all(xval.real == aesara.function([x], real(x))(xval))
        assert np.all(xval.imag == aesara.function([x], imag(x))(xval))

    def test_on_real_input(self):
        x = dvector()
        rng = np.random.default_rng(23)
        xval = rng.standard_normal(10)
        np.all(0 == aesara.function([x], imag(x))(xval))
        np.all(xval == aesara.function([x], real(x))(xval))

        x = imatrix()
        xval = np.asarray(rng.standard_normal((3, 3)) * 100, dtype="int32")
        np.all(0 == aesara.function([x], imag(x))(xval))
        np.all(xval == aesara.function([x], real(x))(xval))

    def test_cast(self):
        x = zvector()
        with pytest.raises(TypeError):
            cast(x, "int32")

    def test_complex(self):
        rng = np.random.default_rng(2333)
        m = fmatrix()
        c = at_complex(m[0], m[1])
        assert c.type == cvector
        r, i = [real(c), imag(c)]
        assert r.type == fvector
        assert i.type == fvector
        f = aesara.function([m], [r, i])

        mval = np.asarray(rng.standard_normal((2, 5)), dtype="float32")
        rval, ival = f(mval)
        assert np.all(rval == mval[0]), (rval, mval[0])
        assert np.all(ival == mval[1]), (ival, mval[1])

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_complex_grads(self):
        def f(m):
            c = at_complex(m[0], m[1])
            return 0.5 * real(c) + 0.9 * imag(c)

        rng = np.random.default_rng(9333)
        mval = np.asarray(rng.standard_normal((2, 5)))
        utt.verify_grad(f, [mval])

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_mul_mixed0(self):
        def f(a):
            ac = at_complex(a[0], a[1])
            return abs((ac) ** 2).sum()

        rng = np.random.default_rng(9333)
        aval = np.asarray(rng.standard_normal((2, 5)))
        try:
            utt.verify_grad(f, [aval])
        except GradientError as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_mul_mixed1(self):
        def f(a):
            ac = at_complex(a[0], a[1])
            return abs(ac).sum()

        rng = np.random.default_rng(9333)
        aval = np.asarray(rng.standard_normal((2, 5)))
        try:
            utt.verify_grad(f, [aval])
        except GradientError as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_mul_mixed(self):
        def f(a, b):
            ac = at_complex(a[0], a[1])
            return abs((ac * b) ** 2).sum()

        rng = np.random.default_rng(9333)
        aval = np.asarray(rng.standard_normal((2, 5)))
        bval = rng.standard_normal(5)
        try:
            utt.verify_grad(f, [aval, bval])
        except GradientError as e:
            print(e.num_grad.gf)
            print(e.analytic_grad)
            raise

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_polar_grads(self):
        def f(m):
            c = complex_from_polar(abs(m[0]), m[1])
            return 0.5 * real(c) + 0.9 * imag(c)

        rng = np.random.default_rng(9333)
        mval = np.asarray(rng.standard_normal((2, 5)))
        utt.verify_grad(f, [mval])

    @pytest.mark.skip(reason="Complex grads not enabled, see #178")
    def test_abs_grad(self):
        def f(m):
            c = at_complex(m[0], m[1])
            return 0.5 * abs(c)

        rng = np.random.default_rng(9333)
        mval = np.asarray(rng.standard_normal((2, 5)))
        utt.verify_grad(f, [mval])

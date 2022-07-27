import numpy as np
import pytest
import scipy.special.cython_special
from numba.types import float32, float64, int32, int64

from aesara.link.numba.dispatch.cython_support import Signature, wrap_cython_function


@pytest.mark.parametrize(
    "sig, expected_result, expected_args",
    [
        (b"double(double)", np.float64, [np.float64]),
        (b"float(unsigned int)", np.float32, [np.uintc]),
        (b"unsigned char(unsigned short foo)", np.ubyte, [np.ushort]),
        (
            b"unsigned char(unsigned short foo, double bar)",
            np.ubyte,
            [np.ushort, np.float64],
        ),
    ],
)
def test_parse_signature(sig, expected_result, expected_args):
    actual = Signature.from_c_types(sig)
    assert actual.res_dtype == expected_result
    assert actual.arg_dtypes == expected_args


@pytest.mark.parametrize(
    "have, want, should_provide",
    [
        (b"double(int)", b"float(int)", True),
        (b"float(int)", b"double(int)", False),
        (b"double(unsigned short)", b"double(unsigned char)", True),
        (b"double(unsigned char)", b"double(short)", False),
        (b"short(double)", b"int(double)", True),
        (b"int(double)", b"short(double)", False),
        (b"float(double, int)", b"float(double, short)", True),
    ],
)
def test_signature_provides(have, want, should_provide):
    have = Signature.from_c_types(have)
    want = Signature.from_c_types(want)
    provides = have.provides(want.res_dtype, want.arg_dtypes)
    assert provides == should_provide


@pytest.mark.parametrize(
    "func, output, inputs, expected",
    [
        (
            scipy.special.cython_special.agm,
            np.float64,
            [np.float64, np.float64],
            float64(float64, float64, int32),
        ),
        (
            scipy.special.cython_special.erfc,
            np.float64,
            [np.float64],
            float64(float64, int32),
        ),
        (
            scipy.special.cython_special.expit,
            np.float32,
            [np.float32],
            float32(float32, int32),
        ),
        (
            scipy.special.cython_special.expit,
            np.float64,
            [np.float64],
            float64(float64, int32),
        ),
        (
            # expn doesn't have a float32 implementation
            scipy.special.cython_special.expn,
            np.float32,
            [np.float32, np.float32],
            float64(float64, float64, int32),
        ),
        (
            # We choose the integer implementation if possible
            scipy.special.cython_special.expn,
            np.float32,
            [np.int64, np.float32],
            float64(int64, float64, int32),
        ),
    ],
)
def test_choose_signature(func, output, inputs, expected):
    wrapper = wrap_cython_function(func, output, inputs)
    assert wrapper.signature() == expected

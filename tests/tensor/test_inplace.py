import numpy as np
import pytest

from tests import unittest_tools as utt
from tests.tensor.utils import (
    _bad_build_broadcast_binary_normal,
    _bad_runtime_broadcast_binary_normal,
    _bad_runtime_inv,
    _good_broadcast_binary_arctan2,
    _good_broadcast_binary_normal,
    _good_broadcast_div_mod_normal_float_inplace,
    _good_broadcast_pow_normal_float_pow,
    _good_broadcast_unary_arccosh,
    _good_broadcast_unary_arcsin_float,
    _good_broadcast_unary_arctanh,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_abs,
    _good_broadcast_unary_normal_float,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_empty_no_complex,
    _good_broadcast_unary_normal_no_complex,
    _good_broadcast_unary_positive_float,
    _good_broadcast_unary_tan,
    _good_broadcast_unary_wide_float,
    _good_inv_inplace,
    _numpy_true_div,
    angle_eps,
    check_floatX,
    copymod,
    div_grad_rtol,
    ignore_isfinite_mode,
    inplace_func,
    makeBroadcastTester,
    upcast_float16_ufunc,
)
from theano import _asarray, config
from theano.scalar.basic import round_half_away_from_zero_vec, upcast
from theano.tensor import vector
from theano.tensor.inplace import (
    abs__inplace,
    add_inplace,
    arccos_inplace,
    arccosh_inplace,
    arcsin_inplace,
    arcsinh_inplace,
    arctan2_inplace,
    arctan_inplace,
    arctanh_inplace,
    ceil_inplace,
    conj_inplace,
    cos_inplace,
    cosh_inplace,
    deg2rad_inplace,
    exp2_inplace,
    exp_inplace,
    expm1_inplace,
    floor_inplace,
    int_div_inplace,
    inv_inplace,
    log1p_inplace,
    log2_inplace,
    log10_inplace,
    log_inplace,
    maximum_inplace,
    minimum_inplace,
    mod_inplace,
    mul_inplace,
    neg_inplace,
    pow_inplace,
    rad2deg_inplace,
    round_half_away_from_zero_inplace,
    round_half_to_even_inplace,
    sgn_inplace,
    sin_inplace,
    sinh_inplace,
    sqr_inplace,
    sqrt_inplace,
    sub_inplace,
    tan_inplace,
    tanh_inplace,
    true_div_inplace,
    trunc_inplace,
    xor_inplace,
)


TestAddInplaceBroadcast = makeBroadcastTester(
    op=add_inplace,
    expected=lambda x, y: x + y,
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    inplace=True,
)

TestSubInplaceBroadcast = makeBroadcastTester(
    op=sub_inplace,
    expected=lambda x, y: x - y,
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    inplace=True,
)

TestMaximumInplaceBroadcast = makeBroadcastTester(
    op=maximum_inplace,
    expected=np.maximum,
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    inplace=True,
)

TestMinimumInplaceBroadcast = makeBroadcastTester(
    op=minimum_inplace,
    expected=np.minimum,
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    inplace=True,
)

TestMulInplaceBroadcast = makeBroadcastTester(
    op=mul_inplace,
    expected=lambda x, y: x * y,
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    inplace=True,
)

TestTrueDivInplaceBroadcast = makeBroadcastTester(
    op=true_div_inplace,
    expected=_numpy_true_div,
    good=copymod(
        _good_broadcast_div_mod_normal_float_inplace,
        # The output is now in float, we cannot work inplace on an int.
        without=["integer", "uint8", "uint16", "int8"],
    ),
    grad_rtol=div_grad_rtol,
    inplace=True,
)

TestInvInplaceBroadcast = makeBroadcastTester(
    op=inv_inplace,
    expected=lambda x: _numpy_true_div(np.int8(1), x),
    good=_good_inv_inplace,
    bad_runtime=_bad_runtime_inv,
    grad_rtol=div_grad_rtol,
    inplace=True,
)

TestModInplaceBroadcast = makeBroadcastTester(
    op=mod_inplace,
    expected=lambda x, y: np.asarray(x % y, dtype=upcast(x.dtype, y.dtype)),
    good=copymod(
        _good_broadcast_div_mod_normal_float_inplace, ["complex1", "complex2"]
    ),
    grad_eps=1e-5,
    inplace=True,
)

TestPowInplaceBroadcast = makeBroadcastTester(
    op=pow_inplace,
    expected=lambda x, y: x ** y,
    good=_good_broadcast_pow_normal_float_pow,
    inplace=True,
    mode=ignore_isfinite_mode,
)

TestNegInplaceBroadcast = makeBroadcastTester(
    op=neg_inplace,
    expected=lambda x: -x,
    good=_good_broadcast_unary_normal,
    inplace=True,
)

TestSgnInplaceBroadcast = makeBroadcastTester(
    op=sgn_inplace,
    expected=np.sign,
    good=_good_broadcast_unary_normal_no_complex,
    inplace=True,
)

TestAbsInplaceBroadcast = makeBroadcastTester(
    op=abs__inplace,
    expected=lambda x: np.abs(x),
    good=_good_broadcast_unary_normal_abs,
    inplace=True,
)

TestIntDivInplaceBroadcast = makeBroadcastTester(
    op=int_div_inplace,
    expected=lambda x, y: check_floatX((x, y), x // y),
    good=_good_broadcast_div_mod_normal_float_inplace,
    # I don't test the grad as the output is always an integer
    # (this is not a continuous output).
    # grad=_grad_broadcast_div_mod_normal,
    inplace=True,
)

TestCeilInplaceBroadcast = makeBroadcastTester(
    op=ceil_inplace,
    expected=upcast_float16_ufunc(np.ceil),
    good=copymod(
        _good_broadcast_unary_normal_no_complex,
        without=["integers", "int8", "uint8", "uint16"],
    ),
    # corner cases includes a lot of integers: points where Ceil is not
    # continuous (not differentiable)
    inplace=True,
)

TestFloorInplaceBroadcast = makeBroadcastTester(
    op=floor_inplace,
    expected=upcast_float16_ufunc(np.floor),
    good=copymod(
        _good_broadcast_unary_normal_no_complex,
        without=["integers", "int8", "uint8", "uint16"],
    ),
    inplace=True,
)

TestTruncInplaceBroadcast = makeBroadcastTester(
    op=trunc_inplace,
    expected=upcast_float16_ufunc(np.trunc),
    good=_good_broadcast_unary_normal_no_complex,
    inplace=True,
)

TestRoundHalfToEvenInplaceBroadcast = makeBroadcastTester(
    op=round_half_to_even_inplace,
    expected=np.round,
    good=_good_broadcast_unary_normal_float_no_complex,
    inplace=True,
)

TestRoundHalfAwayFromZeroInplaceBroadcast = makeBroadcastTester(
    op=round_half_away_from_zero_inplace,
    expected=lambda a: round_half_away_from_zero_vec(a),
    good=_good_broadcast_unary_normal_float_no_empty_no_complex,
    inplace=True,
)

TestSqrInplaceBroadcast = makeBroadcastTester(
    op=sqr_inplace,
    expected=np.square,
    good=_good_broadcast_unary_normal,
    inplace=True,
)

TestExpInplaceBroadcast = makeBroadcastTester(
    op=exp_inplace,
    expected=np.exp,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestExp2InplaceBroadcast = makeBroadcastTester(
    op=exp2_inplace,
    expected=np.exp2,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestExpm1InplaceBroadcast = makeBroadcastTester(
    op=expm1_inplace,
    expected=np.expm1,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestLogInplaceBroadcast = makeBroadcastTester(
    op=log_inplace,
    expected=np.log,
    good=_good_broadcast_unary_positive_float,
    inplace=True,
)

TestLog2InplaceBroadcast = makeBroadcastTester(
    op=log2_inplace,
    expected=np.log2,
    good=_good_broadcast_unary_positive_float,
    inplace=True,
)

TestLog10InplaceBroadcast = makeBroadcastTester(
    op=log10_inplace,
    expected=np.log10,
    good=_good_broadcast_unary_positive_float,
    inplace=True,
)

TestLog1pInplaceBroadcast = makeBroadcastTester(
    op=log1p_inplace,
    expected=np.log1p,
    good=_good_broadcast_unary_positive_float,
    inplace=True,
)

TestSqrtInplaceBroadcast = makeBroadcastTester(
    op=sqrt_inplace,
    expected=np.sqrt,
    good=_good_broadcast_unary_positive_float,
    inplace=True,
)

TestDeg2radInplaceBroadcast = makeBroadcastTester(
    op=deg2rad_inplace,
    expected=np.deg2rad,
    good=_good_broadcast_unary_normal_float_no_complex,
    inplace=True,
    eps=angle_eps,
)

TestRad2degInplaceBroadcast = makeBroadcastTester(
    op=rad2deg_inplace,
    expected=np.rad2deg,
    good=_good_broadcast_unary_normal_float_no_complex,
    inplace=True,
    eps=angle_eps,
)

TestSinInplaceBroadcast = makeBroadcastTester(
    op=sin_inplace,
    expected=np.sin,
    good=_good_broadcast_unary_wide_float,
    inplace=True,
)

TestArcsinInplaceBroadcast = makeBroadcastTester(
    op=arcsin_inplace,
    expected=np.arcsin,
    good=_good_broadcast_unary_arcsin_float,
    inplace=True,
)

TestCosInplaceBroadcast = makeBroadcastTester(
    op=cos_inplace,
    expected=np.cos,
    good=_good_broadcast_unary_wide_float,
    inplace=True,
)

TestArccosInplaceBroadcast = makeBroadcastTester(
    op=arccos_inplace,
    expected=np.arccos,
    good=_good_broadcast_unary_arcsin_float,
    inplace=True,
)

TestTanInplaceBroadcast = makeBroadcastTester(
    op=tan_inplace,
    expected=np.tan,
    good=copymod(
        _good_broadcast_unary_tan, without=["integers", "int8", "uint8", "uint16"]
    ),
    inplace=True,
)

TestArctanInplaceBroadcast = makeBroadcastTester(
    op=arctan_inplace,
    expected=np.arctan,
    good=_good_broadcast_unary_wide_float,
    inplace=True,
)

TestArctan2InplaceBroadcast = makeBroadcastTester(
    op=arctan2_inplace,
    expected=np.arctan2,
    good=copymod(
        _good_broadcast_binary_arctan2,
        without=["integers", "int8", "uint8", "uint16", "dtype_mixup_2"],
    ),
    inplace=True,
)

TestCoshInplaceBroadcast = makeBroadcastTester(
    op=cosh_inplace,
    expected=np.cosh,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestArccoshInplaceBroadcast = makeBroadcastTester(
    op=arccosh_inplace,
    expected=np.arccosh,
    good=copymod(_good_broadcast_unary_arccosh, without=["integers", "uint8"]),
    inplace=True,
)

TestSinhInplaceBroadcast = makeBroadcastTester(
    op=sinh_inplace,
    expected=np.sinh,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestArcsinhInplaceBroadcast = makeBroadcastTester(
    op=arcsinh_inplace,
    expected=np.arcsinh,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestTanhInplaceBroadcast = makeBroadcastTester(
    op=tanh_inplace,
    expected=np.tanh,
    good=_good_broadcast_unary_normal_float,
    inplace=True,
)

TestArctanhInplaceBroadcast = makeBroadcastTester(
    op=arctanh_inplace,
    expected=np.arctanh,
    good=copymod(
        _good_broadcast_unary_arctanh, without=["integers", "int8", "uint8", "uint16"]
    ),
    inplace=True,
)

TestConjInplaceBroadcast = makeBroadcastTester(
    op=conj_inplace,
    expected=np.conj,
    good=_good_broadcast_unary_normal,
    inplace=True,
)


@pytest.mark.xfail(
    config.cycle_detection == "fast" and config.mode != "FAST_COMPILE",
    reason="Cycle detection is fast and mode is FAST_COMPILE",
)
def test_exp_inplace_grad_1():
    utt.verify_grad(
        exp_inplace,
        [
            np.asarray(
                [
                    [1.5089518, 1.48439076, -4.7820262],
                    [2.04832468, 0.50791564, -1.58892269],
                ]
            )
        ],
    )


def test_XOR_inplace():
    dtype = [
        "int8",
        "int16",
        "int32",
        "int64",
    ]

    for dtype in dtype:
        x, y = vector(dtype=dtype), vector(dtype=dtype)
        l = _asarray([0, 0, 1, 1], dtype=dtype)
        r = _asarray([0, 1, 0, 1], dtype=dtype)
        ix = x
        ix = xor_inplace(ix, y)
        gn = inplace_func([x, y], ix)
        _ = gn(l, r)
        # test the in-place stuff
        assert np.all(l == np.asarray([0, 1, 1, 0])), l

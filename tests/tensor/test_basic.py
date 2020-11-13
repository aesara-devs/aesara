import builtins
import itertools
import operator
import warnings
from copy import copy, deepcopy
from functools import partial, reduce
from tempfile import mkstemp

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from tests.tensor.utils import (
    ALL_DTYPES,
    COMPLEX_DTYPES,
    REAL_DTYPES,
    _bad_build_broadcast_binary_normal,
    _bad_runtime_broadcast_binary_normal,
    _bad_runtime_inv,
    _eps,
    _good_broadcast_binary_arctan2,
    _good_broadcast_binary_normal,
    _good_broadcast_div_mod_normal_float,
    _good_broadcast_div_mod_normal_float_no_complex,
    _good_broadcast_pow_normal_float,
    _good_broadcast_unary_arccosh,
    _good_broadcast_unary_arcsin,
    _good_broadcast_unary_arctanh,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_empty_no_complex,
    _good_broadcast_unary_normal_no_complex,
    _good_broadcast_unary_positive,
    _good_broadcast_unary_tan,
    _good_broadcast_unary_wide,
    _good_inv,
    _grad_broadcast_binary_normal,
    _grad_broadcast_pow_normal,
    _grad_broadcast_unary_normal,
    _grad_broadcast_unary_normal_no_complex,
    _grad_broadcast_unary_normal_no_complex_no_corner_case,
    _grad_broadcast_unary_normal_noint,
    _grad_inv,
    _numpy_true_div,
    angle_eps,
    check_floatX,
    copymod,
    div_grad_rtol,
    eval_outputs,
    get_numeric_types,
    ignore_isfinite_mode,
    inplace_func,
    makeBroadcastTester,
    makeTester,
    multi_dtype_cast_checks,
    multi_dtype_checks,
    rand,
    rand_nonzero,
    rand_of_dtype,
    rand_ranged,
    randcomplex,
    randint,
    randint_ranged,
    randuint32,
    upcast_float16_ufunc,
    upcast_int8_nfunc,
)
from theano import change_flags, compile, config, function, gof, shared
from theano.compile import DeepCopyOp
from theano.compile.mode import get_default_mode
from theano.gof.graph import Variable
from theano.scalar import autocast_float, autocast_float_as
from theano.tensor import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    Argmax,
    Choose,
    DimShuffle,
    Dot,
    ExtractDiag,
    Eye,
    Join,
    MaxAndArgmax,
    Mean,
    NoneConst,
    PermuteRowElements,
    Reshape,
    ScalarFromTensor,
    Shape,
    Split,
    Tensor,
    TensorFromScalar,
    TensorType,
    Tile,
    Tri,
    add,
    addbroadcast,
    allclose,
    alloc,
    arange,
    argmax,
    argmin,
    as_tensor_variable,
    batched_dot,
    bvector,
    choose,
    clip,
    constant,
    cscalar,
    default,
    dense_dot,
    diag,
    dmatrix,
    dscalar,
    dscalars,
    dtensor3,
    dtensor4,
    dvector,
    eq,
    exp,
    extract_constant,
    eye,
    fill,
    flatnonzero,
    flatten,
    fmatrix,
    fscalar,
    fscalars,
    fvector,
    get_scalar_constant_value,
    get_vector_length,
    grad,
    hessian,
    horizontal_stack,
    imatrix,
    inverse_permutation,
    iscalar,
    iscalars,
    isclose,
    itensor3,
    ivector,
    join,
    lscalar,
    lvector,
    matrices,
    matrix,
    max,
    max_and_argmax,
    maximum,
    mgrid,
    min,
    minimum,
    mul,
    neq,
    nonzero,
    nonzero_values,
    numeric_grad,
    ogrid,
    opt,
    patternbroadcast,
    permute_row_elements,
    power,
    ptp,
    reshape,
    roll,
    row,
    scal,
    scalar,
    scalar_from_tensor,
    scalars,
    second,
    shape,
    smallest,
    stack,
    stacklists,
    sub,
    sum,
    swapaxes,
    switch,
    tensor3,
    tensor4,
    tensor_copy,
    tensor_from_scalar,
    tensordot,
    tile,
    tri,
    tril,
    triu,
    true_div,
    unbroadcast,
    var,
    vector,
    vertical_stack,
    wvector,
    zvector,
)


if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()

TestAddBroadcast = makeBroadcastTester(
    op=add,
    expected=lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x + y, inputs)),
    good=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        three_inputs_same_shapes_uint=(
            randuint32(2, 3),
            randuint32(2, 3),
            randuint32(2, 3),
        ),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_good_broadcast_binary_normal,
    ),
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
)


TestSubBroadcast = makeBroadcastTester(
    op=sub,
    expected=lambda x, y: check_floatX((x, y), x - y),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)

TestSwitchBroadcast = makeBroadcastTester(
    op=switch,
    expected=np.where,
    good=dict(
        all_true=(np.asarray(1, dtype=config.floatX), rand(4, 5), rand(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), rand(4, 5), rand(4, 5)),
        mixed=(randint_ranged(0, 1, (4, 5)), rand(4, 5), rand(4, 5)),
    ),
    bad_build=dict(all_true=(np.asarray(1, dtype=config.floatX), rand(4, 5))),
    bad_runtime=dict(
        all_true=(np.asarray(1, dtype=config.floatX), rand(3, 5), rand(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), rand(4, 6), rand(4, 5)),
    ),
    # We suppose that cond+eps do not switch branch in switch.grad()
    # So we can't call verify_grad with cond 0.
    grad=dict(
        all_true=(np.asarray(1, dtype=config.floatX), rand(4, 5), rand(4, 5)),
        # false_true=(np.asarray(0, dtype=config.floatX),
        #             rand(4, 5), rand(4, 5)),
        # mixed=(randint_ranged(0, 1, (4, 5)).astype(config.floatX),
        #        rand(4, 5), rand(4, 5))
    ),
)


TestMaximumBroadcast = makeBroadcastTester(
    op=maximum,
    expected=lambda *inputs: check_floatX(inputs, np.maximum(*inputs)),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)


def test_maximum_minimum_grad():
    # Test the discontinuity point.
    # We decided that we only pass the gradient to the first input in that case.
    x, y = tt.vectors("xy")
    for op in [tt.maximum, tt.minimum]:
        o = op(x, y)
        g = theano.grad(o.sum(), [x, y])

        f = theano.function([x, y], g)
        assert np.allclose(f([1], [1]), [[1], [0]])


TestMinimumBroadcast = makeBroadcastTester(
    op=minimum,
    expected=lambda *inputs: check_floatX(inputs, np.minimum(*inputs)),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)

TestMulBroadcast = makeBroadcastTester(
    op=mul,
    expected=lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x * y, inputs)),
    good=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_good_broadcast_binary_normal,
    ),
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_grad_broadcast_binary_normal,
    ),
)

_grad_broadcast_div_mod_normal = dict(
    same_shapes=(rand(2, 3), rand_nonzero((2, 3))),
    scalar=(rand(2, 3), rand_nonzero((1, 1))),
    row=(rand(2, 3), rand_nonzero((1, 3))),
    column=(rand(2, 3), rand_nonzero((2, 1))),
    # complex1=(randcomplex(2, 3), randcomplex_nonzero((2, 3))),
    # complex2=(randcomplex(2, 3), rand_nonzero((2, 3))),
    # complex3=(rand(2, 3), randcomplex_nonzero((2, 3))),
    # dtype_mixup_1=(rand(2, 3), randint_nonzero(2, 3)),
    # dtype_mixup_2=(randint_nonzero(2, 3), rand_nonzero((2, 3))),
    # empty1=(np.asarray([]), np.asarray([1.])),
    # empty2=(np.asarray([0]), np.asarray([])),
)

TestTrueDivBroadcast = makeBroadcastTester(
    op=tt.true_div,
    expected=_numpy_true_div,
    good=_good_broadcast_div_mod_normal_float_no_complex,
    grad=_grad_broadcast_div_mod_normal,
    grad_rtol=div_grad_rtol,
)

TestInvBroadcast = makeBroadcastTester(
    op=tt.inv,
    expected=lambda x: upcast_int8_nfunc(np.true_divide)(np.int8(1), x),
    good=_good_inv,
    bad_runtime=_bad_runtime_inv,
    grad=_grad_inv,
    grad_rtol=div_grad_rtol,
)

TestCeilIntDivBroadcast = makeBroadcastTester(
    op=tt.ceil_intdiv,
    expected=lambda x, y: check_floatX((x, y), (x // y) + ((x % y) != 0)),
    good=_good_broadcast_div_mod_normal_float_no_complex,
    name="CeilIntDiv",
    # As we implement this function with neq, the gradient returned is always 0.
    # grad=_grad_broadcast_div_mod_normal,
    # grad_rtol=div_grad_rtol,
)

TestModBroadcast = makeBroadcastTester(
    op=tt.mod,
    expected=lambda x, y: np.asarray(
        x % y, dtype=theano.scalar.basic.upcast(x.dtype, y.dtype)
    ),
    good=copymod(_good_broadcast_div_mod_normal_float, ["complex1", "complex2"]),
    grad=_grad_broadcast_div_mod_normal,
    grad_eps=1e-5,
)

# Disable NAN checking for pow operator per issue #1780
TestPowBroadcast = makeBroadcastTester(
    op=pow,
    expected=lambda x, y: check_floatX((x, y), x ** y),
    good=_good_broadcast_pow_normal_float,
    grad=_grad_broadcast_pow_normal,
    name="Pow",
    mode=ignore_isfinite_mode,
)

TestAbsBroadcast = makeBroadcastTester(
    op=tt.abs_,
    expected=lambda x: abs(x),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestNegBroadcast = makeBroadcastTester(
    op=tt.neg,
    expected=lambda x: -x,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestSgnBroadcast = makeBroadcastTester(
    op=tt.sgn,
    expected=np.sign,
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal,
)

TestCeilBroadcast = makeBroadcastTester(
    op=tt.ceil,
    expected=upcast_float16_ufunc(np.ceil),
    good=_good_broadcast_unary_normal_no_complex,
    grad=copymod(
        _grad_broadcast_unary_normal_noint,
        extra=[np.asarray([-2.5, -1.5, -1.51, 0.49, 0.98, 1.02], dtype=config.floatX)],
    ),
)

TestFloorBroadcast = makeBroadcastTester(
    op=tt.floor,
    expected=upcast_float16_ufunc(np.floor),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_noint,
)

TestTruncBroadcast = makeBroadcastTester(
    op=tt.trunc,
    expected=upcast_float16_ufunc(np.trunc),
    good=_good_broadcast_unary_normal_no_complex,
)

TestRoundHalfToEvenBroadcast = makeBroadcastTester(
    op=tt.round_half_to_even,
    expected=np.round,
    good=_good_broadcast_unary_normal_float_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex_no_corner_case,
)

# np.vectorize don't handle correctly empty ndarray.
# see in their file numpy/lib/function_base.py in class vectorize.__call__
# This happen in float32 mode.
TestRoundHalfAwayFromZeroBroadcast = makeBroadcastTester(
    op=tt.round_half_away_from_zero,
    expected=lambda a: theano.scalar.basic.round_half_away_from_zero_vec(a),
    good=_good_broadcast_unary_normal_float_no_empty_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex_no_corner_case,
)

TestSqrBroadcast = makeBroadcastTester(
    op=tt.sqr,
    expected=np.square,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestExpBroadcast = makeBroadcastTester(
    op=tt.exp,
    expected=upcast_float16_ufunc(np.exp),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-127, 89, dtype="int8")],
        uint8=[np.arange(0, 89, dtype="uint8")],
        uint16=[np.arange(0, 89, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

TestExp2Broadcast = makeBroadcastTester(
    op=tt.exp2,
    expected=upcast_float16_ufunc(np.exp2),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestExpm1Broadcast = makeBroadcastTester(
    op=tt.expm1,
    expected=upcast_float16_ufunc(np.expm1),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-127, 89, dtype="int8")],
        uint8=[np.arange(0, 89, dtype="uint8")],
        uint16=[np.arange(0, 89, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)


_grad_broadcast_unary_positive = dict(
    normal=(rand_ranged(_eps, 5, (2, 3)),),
)

TestLogBroadcast = makeBroadcastTester(
    op=tt.log,
    expected=upcast_float16_ufunc(np.log),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog2Broadcast = makeBroadcastTester(
    op=tt.log2,
    expected=upcast_float16_ufunc(np.log2),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog10Broadcast = makeBroadcastTester(
    op=tt.log10,
    expected=upcast_float16_ufunc(np.log10),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog1pBroadcast = makeBroadcastTester(
    op=tt.log1p,
    expected=upcast_float16_ufunc(np.log1p),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestSqrtBroadcast = makeBroadcastTester(
    op=tt.sqrt,
    expected=upcast_float16_ufunc(np.sqrt),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

_grad_broadcast_unary_wide = dict(
    normal=(rand_ranged(-1000, 1000, (2, 3)),),
)

TestDeg2radBroadcast = makeBroadcastTester(
    op=tt.deg2rad,
    expected=upcast_float16_ufunc(np.deg2rad),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex,
    eps=angle_eps,
)

TestRad2degBroadcast = makeBroadcastTester(
    op=tt.rad2deg,
    expected=upcast_float16_ufunc(np.rad2deg),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex,
    eps=angle_eps,
)

TestSinBroadcast = makeBroadcastTester(
    op=tt.sin,
    expected=upcast_float16_ufunc(np.sin),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)

# The actual range is [-1, 1] but the numerical gradient is too
# unstable near those values
_grad_broadcast_unary_arcsin = dict(
    normal=(rand_ranged(-0.9, 0.9, (2, 3)),),
)

TestArcsinBroadcast = makeBroadcastTester(
    op=tt.arcsin,
    expected=upcast_float16_ufunc(np.arcsin),
    good=_good_broadcast_unary_arcsin,
    grad=_grad_broadcast_unary_arcsin,
)

TestCosBroadcast = makeBroadcastTester(
    op=tt.cos,
    expected=upcast_float16_ufunc(np.cos),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)


def test_py_c_match():
    a = tt.TensorType(dtype="int8", broadcastable=(False,))()
    f = theano.function([a], tt.arccos(a), mode="DebugMode")
    # This can fail in DebugMode
    f(np.asarray([1, 0, -1], dtype="int8"))


TestArccosBroadcast = makeBroadcastTester(
    op=tt.arccos,
    expected=upcast_float16_ufunc(np.arccos),
    good=_good_broadcast_unary_arcsin,
    grad=_grad_broadcast_unary_arcsin,
)

# We do not want to test around the discontinuity.
_grad_broadcast_unary_tan = dict(
    normal=(rand_ranged(-1.5, 1.5, (2, 3)),), shifted=(rand_ranged(1.6, 4.6, (2, 3)),)
)

TestTanBroadcast = makeBroadcastTester(
    op=tt.tan,
    expected=upcast_float16_ufunc(np.tan),
    good=_good_broadcast_unary_tan,
    grad=_grad_broadcast_unary_tan,
)

TestArctanBroadcast = makeBroadcastTester(
    op=tt.arctan,
    expected=upcast_float16_ufunc(np.arctan),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)

_grad_broadcast_binary_arctan2 = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
)

TestArctan2Broadcast = makeBroadcastTester(
    op=tt.arctan2,
    expected=upcast_float16_ufunc(np.arctan2),
    good=_good_broadcast_binary_arctan2,
    grad=_grad_broadcast_binary_arctan2,
)

TestCoshBroadcast = makeBroadcastTester(
    op=tt.cosh,
    expected=upcast_float16_ufunc(np.cosh),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-89, 90, dtype="int8")],
        uint8=[np.arange(0, 90, dtype="uint8")],
        uint16=[np.arange(0, 90, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

_grad_broadcast_unary_arccosh = dict(
    normal=(rand_ranged(1 + _eps, 1000, (2, 3)),),
)

TestArccoshBroadcast = makeBroadcastTester(
    op=tt.arccosh,
    expected=upcast_float16_ufunc(np.arccosh),
    good=_good_broadcast_unary_arccosh,
    grad=_grad_broadcast_unary_arccosh,
)

TestSinhBroadcast = makeBroadcastTester(
    op=tt.sinh,
    expected=upcast_float16_ufunc(np.sinh),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-89, 90, dtype="int8")],
        uint8=[np.arange(0, 90, dtype="uint8")],
        uint16=[np.arange(0, 90, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

TestArcsinhBroadcast = makeBroadcastTester(
    op=tt.arcsinh,
    expected=upcast_float16_ufunc(np.arcsinh),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestTanhBroadcast = makeBroadcastTester(
    op=tt.tanh,
    expected=upcast_float16_ufunc(np.tanh),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

_grad_broadcast_unary_arctanh = dict(
    normal=(rand_ranged(-1 + _eps, 1 - _eps, (2, 3)),),
)

TestArctanhBroadcast = makeBroadcastTester(
    op=tt.arctanh,
    expected=upcast_float16_ufunc(np.arctanh),
    good=_good_broadcast_unary_arctanh,
    grad=_grad_broadcast_unary_arctanh,
)

TestZerosLikeBroadcast = makeBroadcastTester(
    op=tt.zeros_like,
    expected=np.zeros_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="ZerosLike",
)

TestOnesLikeBroadcast = makeBroadcastTester(
    op=tt.ones_like,
    expected=np.ones_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="OnesLike",
)

# Complex operations
_good_complex_from_polar = dict(
    same_shapes=(abs(rand(2, 3)), rand(2, 3)),
    not_same_dimensions=(abs(rand(2, 2)), rand(2)),
    scalar=(abs(rand(2, 3)), rand(1, 1)),
    row=(abs(rand(2, 3)), rand(1, 3)),
    column=(abs(rand(2, 3)), rand(2, 1)),
    integers=(abs(randint(2, 3)), randint(2, 3)),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([1], dtype=config.floatX)),
)
_grad_complex_from_polar = dict(
    same_shapes=(abs(rand(2, 3)), rand(2, 3)),
    scalar=(abs(rand(2, 3)), rand(1, 1)),
    row=(abs(rand(2, 3)), rand(1, 3)),
    column=(abs(rand(2, 3)), rand(2, 1)),
)

TestComplexFromPolarBroadcast = makeBroadcastTester(
    op=tt.complex_from_polar,
    expected=lambda r, theta: r * np.cos(theta) + 1j * r * np.sin(theta),
    good=_good_complex_from_polar,
)

TestConjBroadcast = makeBroadcastTester(
    op=tt.conj, expected=np.conj, good=_good_broadcast_unary_normal
)


TestDenseDot = makeTester(
    name="DenseDotTester",
    op=dense_dot,
    expected=lambda x, y: np.dot(x, y),
    checks={},
    good=dict(
        correct1=(rand(5, 7), rand(7, 5)),
        correct2=(rand(5, 7), rand(7, 9)),
        correct3=(rand(5, 7), rand(7)),
        correct4=(rand(5), rand(5, 7)),
        mixed1=(rand(5).astype("float32"), rand(5, 7)),
        mixed2=(rand(5).astype("float64"), rand(5, 7)),
        complex1=(randcomplex(5, 7), randcomplex(7)),
        complex2=(rand(5, 7), randcomplex(7)),
        complex3=(randcomplex(5, 7), rand(7)),
        empty1=(
            np.asarray([], dtype=config.floatX),
            np.asarray([], dtype=config.floatX),
        ),
        empty2=(rand(5, 0), rand(0, 2)),
        empty3=(rand(0, 5), rand(5, 0)),
    ),
    bad_build=dict(),
    bad_runtime=dict(bad1=(rand(5, 7), rand(5, 7)), bad2=(rand(5, 7), rand(8, 3))),
)

TestBatchedDot = makeTester(
    name="BatchedDotTester",
    op=batched_dot,
    expected=(
        lambda xs, ys: np.asarray(
            list(
                x * y if x.ndim == 0 or y.ndim == 0 else np.dot(x, y)
                for x, y in zip(xs, ys)
            ),
            dtype=theano.scalar.upcast(xs.dtype, ys.dtype),
        )
    ),
    checks={},
    grad=dict(
        correct1=(rand(3, 5, 7), rand(3, 7, 5)),
        correct2=(rand(3, 5, 7), rand(3, 7, 9)),
        correct3=(rand(3, 5, 7), rand(3, 7)),
        correct4=(rand(3, 5), rand(3, 5, 7)),
        correct5=(rand(3), rand(3, 5, 7)),
        correct6=(rand(3, 5), rand(3)),
        correct7=(rand(3, 5), rand(3, 5)),
        correct8=(rand(3), rand(3)),
        correct9=(rand(3, 5, 7, 11), rand(3)),
        correct10=(rand(3, 2, 6, 5), rand(3, 5)),
        correct11=(rand(3, 2, 6, 5), rand(3, 5, 7)),
        correct12=(rand(3, 2, 6, 5), rand(3, 7, 5, 8)),
        mixed1=(rand(3, 5).astype("float32"), rand(3, 5, 7)),
        mixed2=(rand(3, 5).astype("float64"), rand(3, 5, 7)),
    ),
    good=dict(
        correct1=(rand(3, 5, 7), rand(3, 7, 5)),
        correct2=(rand(3, 5, 7), rand(3, 7, 9)),
        correct3=(rand(3, 5, 7), rand(3, 7)),
        correct4=(rand(3, 5), rand(3, 5, 7)),
        correct5=(rand(3), rand(3, 5, 7)),
        correct6=(rand(3, 5), rand(3)),
        correct7=(rand(3, 5), rand(3, 5)),
        correct8=(rand(3), rand(3)),
        correct9=(rand(3, 5, 7, 11), rand(3)),
        correct10=(rand(3, 7, 11, 5), rand(3, 5)),
        correct11=(rand(3, 7, 11, 5), rand(3, 5, 13)),
        correct12=(rand(3, 7, 11, 5), rand(3, 13, 5, 17)),
        mixed1=(rand(3, 5).astype("float32"), rand(3, 5, 7)),
        mixed2=(rand(3, 5).astype("float64"), rand(3, 5, 7)),
    ),
    bad_build=dict(
        no_batch_axis2=(rand(), rand(3, 5)), no_batch_axis3=(rand(3, 5), rand())
    ),
    bad_runtime=dict(
        batch_dim_mismatch1=(rand(2, 5, 7), rand(3, 7, 9)),
        batch_dim_mismatch2=(rand(3, 5, 7), rand(2, 7, 9)),
        batch_dim_mismatch3=(rand(3), rand(5)),
        bad_dim1=(rand(3, 5, 7), rand(3, 5, 7)),
        bad_dim2=(rand(3, 5, 7), rand(3, 8, 3)),
        bad_dim3=(rand(3, 5), rand(3, 7)),
        bad_dim4=(rand(3, 5, 7, 11), rand(3, 5)),
        bad_dim5=(rand(3, 5, 7, 11), rand(3, 5, 13)),
        bad_dim6=(rand(3, 5, 7, 11), rand(3, 13, 5, 17)),
    ),
)


def _numpy_second(x, y):
    return np.broadcast_arrays(x, y)[1]


TestSecondBroadcast = makeTester(
    name="SecondBroadcastTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5,)),
            multi_dtype_checks((2, 3, 2), (3, 2)),
            multi_dtype_checks((2, 3, 2), (2,)),
        )
    ),
    # I can't think of any way to make this fail at build time
    # Just some simple smoke tests
    bad_runtime=dict(
        fail1=(rand(5, 4), rand(5)),
        fail2=(rand(3, 2, 3), rand(6, 9)),
        fail3=(randint(6, 2, 9), rand(3, 2)),
    ),
)

# We exclude local_fill_to_alloc because it optimizes the "second" node
# away from the graph.
TestSecondSameRank = makeTester(
    name="SecondSameRankTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (4, 5)),
            multi_dtype_checks((1, 2), (3, 2)),
            multi_dtype_checks((3, 2), (1, 2)),
        )
    ),
    # These sizes are not broadcastable to one another
    # and SHOULD raise an error, but currently don't.
    bad_runtime=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5, 4)),
            multi_dtype_checks((1, 5), (5, 4)),
        )
    ),
    mode=get_default_mode().excluding("local_fill_to_alloc", "local_useless_fill"),
)

# Alloc
TestAllocBroadcast = makeBroadcastTester(
    name="AllocTester",
    op=alloc,
    expected=(lambda x, *shp: np.zeros(shp, dtype=x.dtype) + x),
    good=dict(
        correct01=(rand(), np.int32(7)),
        correct01_bcast=(rand(1), np.int32(7)),
        correct02=(rand(), np.int32(4), np.int32(7)),
        correct12=(rand(7), np.int32(4), np.int32(7)),
        correct13=(rand(7), np.int32(2), np.int32(4), np.int32(7)),
        correct23=(rand(4, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb1=(rand(1, 7), np.int32(4), np.int32(7)),
        correctb2=(rand(1, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb3=(rand(7, 1), np.int32(7), np.int32(4)),
        correctb4=(rand(7, 1), np.int32(2), np.int32(7), np.int32(4)),
    ),
    bad_runtime=dict(
        bad_shape12=(rand(7), np.int32(7), np.int32(5)),
    ),
    bad_build=dict(
        vec=(rand(1), [np.int32(2)]),
        too_big32=(rand(6, 2, 4), np.int32(6), np.int32(2)),
        too_big32b=(rand(6, 2, 4), np.int32(6), np.int32(4)),
        too_big32c=(rand(6, 2, 4), np.int32(2), np.int32(4)),
        too_big32d=(rand(6, 2, 4), np.int32(2), np.int32(6)),
        too_big32e=(rand(6, 2, 4), np.int32(4), np.int32(6)),
        too_big32f=(rand(6, 2, 4), np.int32(4), np.int32(2)),
    ),
)

# Since not all inputs of Alloc are differentiable, we need different testers
s1, s2, s3 = randint_ranged(1, 13, (3,))
# alloc a scalar into a vector
TestAlloc01GradBroadcast = makeBroadcastTester(
    name="Alloc01GradTester",
    op=(lambda x: alloc(x, s1)),
    expected=(lambda x: np.zeros((s1,), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(),),
        x2=(rand(),),
        x3=(rand(),),
    ),
)

# alloc a vector into a tensor3
TestAlloc13GradBroadcast = makeBroadcastTester(
    name="Alloc13GradTester",
    op=(lambda x: alloc(x, s1, s2, s3)),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(s3),),
        x2=(rand(s3),),
        x3=(rand(s3),),
    ),
)

# unbroadcast a row to a matrix
TestAllocb1GradBroadcast = makeBroadcastTester(
    name="Allocb1GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(1, s2),),
        x2=(rand(1, s2),),
        x3=(rand(1, s2),),
    ),
)

# unbroadcast a row to a tensor3
TestAllocb2GradBroadcast = makeBroadcastTester(
    name="Allocb2GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(1, s3),),
        x2=(rand(1, s3),),
        x3=(rand(1, s3),),
    ),
)

# unbroadcast a col to a matrix
TestAllocb3GradBroadcast = makeBroadcastTester(
    name="Allocb3GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(s1, 1),),
        x2=(rand(s1, 1),),
        x3=(rand(s1, 1),),
    ),
)

# unbroadcast a col to a tensor3
TestAllocb4GradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(s2, 1),),
        x2=(rand(s2, 1),),
        x3=(rand(s2, 1),),
    ),
)


# Partial un broadcast of a dimshuffled input
TestAllocDimshuffleGradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", "x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(s3),),
        x2=(rand(s3),),
        x3=(rand(s3),),
    ),
)
TestAllocDimshuffleGrad2Broadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(rand(s3),),
        x2=(rand(s3),),
        x3=(rand(s3),),
    ),
)


class ApplyDefaultTestOp(theano.Op):
    def __init__(self, id):
        self.default_output = id

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])


def test_constant():
    int8_vector_type = tt.TensorType(dtype="int8", broadcastable=(False,))

    # Make sure we return a `TensorConstant` unchanged
    x = tt.TensorConstant(int8_vector_type, [1, 2])
    y = constant(x)
    assert y is x

    # Make sure we can add and remove broadcastable dimensions
    int8_scalar_type = tt.TensorType(dtype="int8", broadcastable=())
    x_data = np.array(2, dtype="int8")

    x = tt.TensorConstant(int8_scalar_type, x_data)
    y = constant(x, ndim=1)
    assert y.ndim == 1
    assert np.array_equal(y.data, np.expand_dims(x_data, 0))

    y = constant(x, ndim=2)
    assert y.ndim == 2
    assert np.array_equal(y.data, np.expand_dims(x_data, (0, 1)))

    z = constant(y, ndim=0)
    assert y.ndim == 2 and z.ndim == 0
    assert np.array_equal(z.data, x_data)


class TestAsTensorVariable:
    """
    Unit test for ensuring that as_tensor_variable handles Apply objects
    correctly and removes leading broadcastable dimensions when possible.
    """

    def setup_method(self):
        self.x = tt.scalar("x")

    def test_tensor_from_scalar(self):
        y = as_tensor_variable(scal.int8())
        assert isinstance(y.owner.op, TensorFromScalar)

    def test_multi_outputs(self):
        good_apply_var = ApplyDefaultTestOp(0).make_node(self.x)
        as_tensor_variable(good_apply_var)

        bad_apply_var = ApplyDefaultTestOp(-1).make_node(self.x)
        with pytest.raises(ValueError):
            _ = as_tensor_variable(bad_apply_var)

        bad_apply_var = ApplyDefaultTestOp(2).make_node(self.x)
        with pytest.raises(ValueError):
            _ = as_tensor_variable(bad_apply_var)

    def test_list(self):
        # Make sure our exception handling during `Sequence` processing doesn't
        # mask exceptions caused by unrelated logic (e.g.  computing test
        # values)
        with change_flags(compute_test_value="raise"), pytest.raises(ValueError):
            a = tt.lscalar("a")
            y = (a, a, 1)
            _ = as_tensor_variable(y)

        bad_apply_var = ApplyDefaultTestOp([0, 1]).make_node(self.x)
        with pytest.raises(ValueError):
            as_tensor_variable(bad_apply_var)

    def test_strip_leading_broadcastable(self):
        x = tt.TensorType(config.floatX, (True, False))("x")
        x = as_tensor_variable(x, ndim=1)
        assert x.ndim == 1

        x = tt.matrix("x", dtype=config.floatX)
        with pytest.raises(ValueError):
            as_tensor_variable(x, ndim=1)

    def test_bool(self):
        # We should not allow `as_tensor_variable` to accept `True` or `False`,
        # but it should up-cast an `ndarray` of `bool` to uint8
        with pytest.raises(TypeError):
            as_tensor_variable(True)

        ten = as_tensor_variable(np.array([True, False, False, True, True]))
        assert ten.type.dtype == "bool"

    def test_memmap(self):
        inp = np.random.rand(4, 3)
        _, fname = mkstemp()
        new_inp = np.memmap(fname, dtype=inp.dtype, mode="w+", shape=inp.shape)
        new_inp[...] = inp
        res = as_tensor_variable(new_inp)
        assert isinstance(res, tt.TensorConstant)
        assert res.data is new_inp

    @pytest.mark.parametrize(
        "dtype",
        [
            "float16",
            "float32",
            "float64",
        ],
    )
    def test_empty_dtype(self, dtype):
        with theano.change_flags(floatX=dtype):
            assert as_tensor_variable(()).dtype == dtype
            assert as_tensor_variable([]).dtype == dtype

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([1, 2], [1, 2]),
            ([tt.as_tensor(1), tt.as_tensor(2)], [1, 2]),
            ([theano.scalar.constant(1), theano.scalar.constant(2)], [1, 2]),
        ],
    )
    def test_constant_consistency(self, x, y):
        a = as_tensor_variable(x)
        assert isinstance(a, tt.TensorConstant)
        assert np.array_equal(a.data, y)

    def test_constant_identity(self):
        # Values that are already `TensorType`s shouldn't be recreated by
        # `as_tensor_variable`
        x_scalar = tt.TensorConstant(tt.TensorType(dtype="int8", broadcastable=()), 2)
        a_scalar = as_tensor_variable(x_scalar)
        assert x_scalar is a_scalar

        x_vector = tt.TensorConstant(
            tt.TensorType(dtype="int8", broadcastable=(False,)),
            np.array([1, 2], dtype="int8"),
        )
        a_vector = as_tensor_variable(x_vector)
        assert x_vector is a_vector


class TestAlloc:
    dtype = config.floatX
    mode = mode_opt
    shared = staticmethod(theano.shared)
    allocs = [tt.Alloc()] * 3

    def setup_method(self):
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_alloc_constant_folding(self):
        test_params = np.asarray(self.rng.randn(50 * 60), self.dtype)

        some_vector = vector("some_vector", dtype=self.dtype)
        some_matrix = some_vector.reshape((60, 50))
        variables = self.shared(np.ones((50,), dtype=self.dtype))
        idx = tt.constant(np.arange(50))

        for alloc_, (subtensor, n_alloc) in zip(
            self.allocs,
            [
                # IncSubtensor1
                (some_matrix[:60], 2),
                # AdvancedIncSubtensor1
                (some_matrix[arange(60)], 2),
                # AdvancedIncSubtensor
                (some_matrix[idx, idx], 1),
            ],
        ):
            derp = sum(dense_dot(subtensor, variables))

            fobj = theano.function([some_vector], derp, mode=self.mode)
            grad_derp = theano.grad(derp, some_vector)
            fgrad = theano.function([some_vector], grad_derp, mode=self.mode)

            topo_obj = fobj.maker.fgraph.toposort()
            assert np.sum([isinstance(node.op, type(alloc_)) for node in topo_obj]) == 0

            topo_grad = fgrad.maker.fgraph.toposort()
            assert (
                np.sum([isinstance(node.op, type(alloc_)) for node in topo_grad])
                == n_alloc
            ), (alloc_, subtensor, n_alloc, topo_grad)
            fobj(test_params)
            fgrad(test_params)

    def test_alloc_output(self):
        val = tt.constant(self.rng.randn(1, 1), dtype=self.dtype)
        for alloc_ in self.allocs:
            # The output is the result of the alloc operation,
            # we do not want it to be constant-folded
            out = alloc_(val, 50, 60)

            f = theano.function([], out, mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert np.sum([isinstance(node.op, type(alloc_)) for node in topo]) == 1
            assert not isinstance(topo[0].op, DeepCopyOp)

    def test_ones(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            ones = theano.function([], [tt.ones(shp)], mode=self.mode)
            assert np.allclose(ones(), np.ones(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        ones_scalar = theano.function([], [tt.ones(x.shape)], mode=self.mode)
        assert np.allclose(ones_scalar(), np.ones(shp))

        for (typ, shp) in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            ones_tensor = theano.function([x], [tt.ones(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(ones_tensor(inp), np.ones(shp))

    def test_zeros(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            zeros = theano.function([], [tt.zeros(shp)], mode=self.mode)
            assert np.allclose(zeros(), np.zeros(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        zeros_scalar = theano.function([], [tt.zeros(x.shape)], mode=self.mode)
        assert np.allclose(zeros_scalar(), np.zeros(shp))

        for (typ, shp) in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            zeros_tensor = theano.function([x], [tt.zeros(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(zeros_tensor(inp), np.zeros(shp))


# This is slow for the ('int8', 3) version.
def test_eye():
    def check(dtype, N, M_=None, k=0):
        # Theano does not accept None as a tensor.
        # So we must use a real value.
        M = M_
        # Currently DebugMode does not support None as inputs even if this is
        # allowed.
        if M is None and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            M = N
        N_symb = tt.iscalar()
        M_symb = tt.iscalar()
        k_symb = tt.iscalar()
        f = function([N_symb, M_symb, k_symb], eye(N_symb, M_symb, k_symb, dtype=dtype))
        result = f(N, M, k)
        assert np.allclose(result, np.eye(N, M_, k, dtype=dtype))
        assert result.dtype == np.dtype(dtype)

    for dtype in ALL_DTYPES:
        check(dtype, 3)
        # M != N, k = 0
        check(dtype, 3, 5)
        check(dtype, 5, 3)
        # N == M, k != 0
        check(dtype, 3, 3, 1)
        check(dtype, 3, 3, -1)
        # N < M, k != 0
        check(dtype, 3, 5, 1)
        check(dtype, 3, 5, -1)
        # N > M, k != 0
        check(dtype, 5, 3, 1)
        check(dtype, 5, 3, -1)


class TestTriangle:
    def test_tri(self):
        def check(dtype, N, M_=None, k=0):
            # Theano does not accept None as a tensor.
            # So we must use a real value.
            M = M_
            # Currently DebugMode does not support None as inputs even if this is
            # allowed.
            if M is None and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
                M = N
            N_symb = tt.iscalar()
            M_symb = tt.iscalar()
            k_symb = tt.iscalar()
            f = function(
                [N_symb, M_symb, k_symb], tri(N_symb, M_symb, k_symb, dtype=dtype)
            )
            result = f(N, M, k)
            assert np.allclose(result, np.tri(N, M_, k, dtype=dtype))
            assert result.dtype == np.dtype(dtype)

        for dtype in ALL_DTYPES:
            check(dtype, 3)
            # M != N, k = 0
            check(dtype, 3, 5)
            check(dtype, 5, 3)
            # N == M, k != 0
            check(dtype, 3, 3, 1)
            check(dtype, 3, 3, -1)
            # N < M, k != 0
            check(dtype, 3, 5, 1)
            check(dtype, 3, 5, -1)
            # N > M, k != 0
            check(dtype, 5, 3, 1)
            check(dtype, 5, 3, -1)

    def test_tril_triu(self):
        def check_l(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], tril(m_symb, k_symb))
            result = f(m, k)
            assert np.allclose(result, np.tril(m, k))
            assert result.dtype == np.dtype(dtype)

        def check_u(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], triu(m_symb, k_symb))
            result = f(m, k)
            assert np.allclose(result, np.triu(m, k))
            assert result.dtype == np.dtype(dtype)

        for dtype in ALL_DTYPES:
            m = rand_of_dtype((10, 10), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)

            m = rand_of_dtype((10, 5), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)


class TestNonzero:
    @change_flags(compute_test_value="raise")
    def test_nonzero(self):
        def check(m):
            m_symb = tt.tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_tuple_tt = nonzero(m_symb, return_matrix=False)
            res_matrix_tt = nonzero(m_symb, return_matrix=True)

            res_tuple = tuple(r.tag.test_value for r in res_tuple_tt)
            res_matrix = res_matrix_tt.tag.test_value

            assert np.allclose(res_matrix, np.vstack(np.nonzero(m)))

            for i, j in zip(res_tuple, np.nonzero(m)):
                assert np.allclose(i, j)

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

    @change_flags(compute_test_value="raise")
    def test_flatnonzero(self):
        def check(m):
            m_symb = tt.tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_tt = flatnonzero(m_symb)

            result = res_tt.tag.test_value
            assert np.allclose(result, np.flatnonzero(m))

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

    @change_flags(compute_test_value="raise")
    def test_nonzero_values(self):
        def check(m):
            m_symb = tt.tensor(dtype=m.dtype, broadcastable=(False,) * m.ndim)
            m_symb.tag.test_value = m

            res_tt = nonzero_values(m_symb)

            result = res_tt.tag.test_value
            assert np.allclose(result, m[np.nonzero(m)])

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)


def test_identity():
    def check(dtype):
        obj = rand_of_dtype((2,), dtype)
        sym = tt.vector(dtype=dtype)
        f = function([sym], tensor_copy(sym))
        assert np.all(obj == f(obj))
        assert obj.dtype == f(obj).dtype
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        if theano.config.mode != "FAST_COMPILE":
            assert isinstance(topo[0].op, DeepCopyOp)

    for dtype in ALL_DTYPES:
        check(dtype)


class TestCast:
    def test_good_between_real_types(self):
        good = itertools.chain(
            multi_dtype_cast_checks((2,), dtypes=REAL_DTYPES),
            # Casts from foo to foo
            [
                (
                    "{}_{}".format(rand_of_dtype((2,), dtype), dtype),
                    (rand_of_dtype((2,), dtype), dtype),
                )
                for dtype in ALL_DTYPES
            ],
        )
        for testname, (obj, dtype) in good:
            inp = tt.vector(dtype=obj.dtype)
            out = tt.cast(inp, dtype=dtype)
            f = function([inp], out)
            assert f(obj).dtype == np.dtype(dtype)

            # Test astype too
            out2 = inp.astype(dtype=dtype)
            assert out2.type == out.type

    def test_cast_from_real_to_complex(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = tt.vector(dtype=real_dtype)
                out = tt.cast(inp, dtype=complex_dtype)
                f = function([inp], out)
                obj = rand_of_dtype((2,), real_dtype)
                assert f(obj).dtype == np.dtype(complex_dtype)

    def test_cast_from_complex_to_real_raises_error(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = tt.vector(dtype=real_dtype)
                with pytest.raises(TypeError):
                    tt.tensor(tt.cast(inp, dtype=complex_dtype))


TestClip = makeTester(
    name="ClipTester",
    op=clip,
    expected=lambda x, y, z: np.clip(x, y, z),
    good=dict(
        correct1=(
            (5 * rand(5, 5)).astype("float32"),
            np.array(-1, dtype="float32"),
            np.array(1, dtype="float32"),
        ),
        correct2=(
            (5 * rand(5, 5)).astype("float64"),
            np.array(-1, dtype="float64"),
            np.array(1, dtype="float64"),
        ),
        correct3=(
            randint(5, 5).astype("int8"),
            np.array(-1, dtype="int8"),
            np.array(1, dtype="int8"),
        ),
        correct4=(
            randint(5, 5).astype("int16"),
            np.array(-1, dtype="int16"),
            np.array(1, dtype="int16"),
        ),
        correct5=(
            randint(5, 5).astype("int32"),
            np.array(-1, dtype="int32"),
            np.array(1, dtype="int32"),
        ),
        correct6=(
            randint(5, 5).astype("int64"),
            np.array(-1, dtype="int64"),
            np.array(1, dtype="int64"),
        ),
        # min > max case moved below as numpy has changed
        correct8=(
            randint(0, 5).astype("uint8"),
            np.array(2, dtype="uint8"),
            np.array(4, dtype="uint8"),
        ),
        correct9=(
            randint(0, 5).astype("uint16"),
            np.array(2, dtype="uint16"),
            np.array(4, dtype="uint16"),
        ),
    )
    # I can't think of any way to make this fail at runtime
)


# min > max case - numpy.clip has changed but we haven't
# https://github.com/Theano/Theano/issues/6715
TestBackwardsClip = makeTester(
    name="BackwardsClipTester",
    op=clip,
    expected=lambda x, y, z: np.where(x < y, y, np.minimum(x, z)),
    good=dict(
        correct7=(
            (5 * rand(5, 5)).astype("float64"),
            np.array(1, dtype="float64"),
            np.array(-1, dtype="float64"),
        ),
    ),
)


class TestClip:
    def test_complex_value(self):
        for dtype in ["complex64", "complex128"]:
            a = tt.vector(dtype=dtype)
            b = tt.scalar()
            c = tt.scalar()
            with pytest.raises(TypeError):
                clip(a, b, c)

    def test_clip_repeat_grad(self):
        # This is testing for the issue #633
        x, y = tt.vectors("xy")
        a = clip(x, y, x)
        g = theano.gradient.grad(a.sum(), x)
        fn = theano.function([x, y], [g])

        # Test the other way around as well
        a2 = clip(x, x, y)
        g2 = theano.gradient.grad(a2.sum(), x)
        fn2 = theano.function([x, y], [g2])

        # Test for the equal case too
        a3 = tt.clip(x, x, x)
        g3 = theano.gradient.grad(a3.sum(), x)
        fn3 = theano.function([x], [g3])

        rng = np.random.RandomState(utt.fetch_seed())

        nvals = 50
        xval = rng.rand(nvals).astype(config.floatX)
        # To ensure that the min < x
        yval_mn = rng.rand(nvals).astype(config.floatX) - 1.0

        # To ensure that the max > x
        yval_mx = rng.rand(nvals).astype(config.floatX) + 1.0

        (aval,) = fn(xval, yval_mn)
        (aval2,) = fn2(xval, yval_mx)
        (aval3,) = fn3(xval)
        assert np.all(aval == 1.0)
        assert np.all(aval2 == 1.0)
        assert np.all(aval3 == 1.0)

    def test_clip_repeat_verify_grad(self):
        # Additional tests for issue gh-633
        utt.verify_grad(op=lambda x: clip(x, 0, x), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(x, x, 0), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(0, x, x), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(x, x, x), pt=[rand_nonzero((3, 7))])


# TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary every time you want to verify
#      gradient numerically


def test_batched_dot():
    first = tt.tensor3("first")
    second = tt.tensor3("second")
    output = tt.basic.batched_dot(first, second)
    first_val = np.random.rand(10, 10, 20).astype(config.floatX)
    second_val = np.random.rand(10, 20, 5).astype(config.floatX)
    result_fn = theano.function([first, second], output)
    result = result_fn(first_val, second_val)
    assert result.shape[0] == first_val.shape[0]
    assert result.shape[1] == first_val.shape[1]
    assert result.shape[2] == second_val.shape[2]

    first_mat = tt.dmatrix("first")
    second_mat = tt.dmatrix("second")
    output = tt.basic.batched_dot(first_mat, second_mat)
    first_mat_val = np.random.rand(10, 10).astype(config.floatX)
    second_mat_val = np.random.rand(10, 10).astype(config.floatX)
    result_fn = theano.function([first_mat, second_mat], output)
    result = result_fn(first_mat_val, second_mat_val)

    assert result.shape[0] == first_mat_val.shape[0]


def test_batched_dot_not_contiguous():
    def np_genarray(*_shape):
        size = 1
        for dimsize in _shape:
            size *= dimsize
        return np.arange(size, dtype=config.floatX).reshape(_shape)

    X = tensor3()
    W = tensor3()
    Z = batched_dot(X, W)
    f = function([X, W], Z)

    w = np_genarray(30, 10, 5)
    reversed_x_container = np_genarray(20, 40, 30)
    x_container = reversed_x_container.T

    def check_first_dim(inverted):
        direction = -1 if inverted else 1
        x = x_container[::direction, ::2, ::2]
        assert x.shape == (30, 20, 10)
        assert x.strides[0] == direction * np.dtype(config.floatX).itemsize
        assert not (x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"])
        result = f(x, w)
        ref_result = np.asarray(list(np.dot(u, v) for u, v in zip(x, w)))
        utt.assert_allclose(ref_result, result)

    for inverted in (0, 1):
        check_first_dim(inverted)


def test_batched_tensordot():
    first = tt.tensor4("first")
    second = tt.tensor4("second")
    axes = [[1, 2], [3, 1]]
    output = tt.batched_tensordot(first, second, axes)
    first_val = np.random.rand(8, 10, 20, 3).astype(config.floatX)
    second_val = np.random.rand(8, 20, 5, 10).astype(config.floatX)
    result_fn = theano.function([first, second], output)
    result = result_fn(first_val, second_val)
    assert result.shape[0] == first_val.shape[0]
    assert result.shape[1] == first_val.shape[3]
    assert result.shape[2] == second_val.shape[2]

    first_mat = tt.dmatrix("first")
    second_mat = tt.dmatrix("second")
    axes = 1
    output = tt.batched_tensordot(first_mat, second_mat, axes)
    first_mat_val = np.random.rand(10, 4).astype(config.floatX)
    second_mat_val = np.random.rand(10, 4).astype(config.floatX)
    result_fn = theano.function([first_mat, second_mat], output)
    result = result_fn(first_mat_val, second_mat_val)
    assert result.shape[0] == first_mat_val.shape[0]
    assert len(result.shape) == 1


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


def test_nan_inf_constant_signature():
    # Test that the signature of a constant tensor containing NaN and Inf
    # values is correct.
    test_constants = [
        [np.nan, np.inf, 0, 1],
        [np.nan, np.inf, -np.inf, 1],
        [0, np.inf, -np.inf, 1],
        [0, 3, -np.inf, 1],
        [0, 3, np.inf, 1],
        [np.nan, 3, 4, 1],
        [0, 3, 4, 1],
        np.nan,
        np.inf,
        -np.inf,
        0,
        1,
    ]
    n = len(test_constants)
    # We verify that signatures of two rows i, j in the matrix above are
    # equal if and only if i == j.
    for i in range(n):
        for j in range(n):
            x = constant(test_constants[i])
            y = constant(test_constants[j])
            assert (x.signature() == y.signature()) == (i == j)

    # Also test that nan !=0 and nan != nan.
    x = tt.scalar()
    mode = get_default_mode()
    if isinstance(mode, theano.compile.debugmode.DebugMode):
        # Disable the check preventing usage of NaN / Inf values.
        # We first do a copy of the mode to avoid side effects on other tests.
        mode = copy(mode)
        mode.check_isfinite = False
    f = theano.function([x], eq(x, np.nan), mode=mode)

    assert f(0) == 0
    assert f(np.nan) == 0


def test_isnan():
    for x in [tt.matrix(), tt.imatrix(), tt.matrix(dtype="bool")]:
        y = tt.isnan(x)
        assert isinstance(y.owner.op, tt.Elemwise) == (
            x.dtype not in tt.discrete_dtypes
        )
        assert y.dtype == "bool"

        # Test c code generator even for int type.
        y = tt.isnan_(x)
        assert isinstance(y.owner.op, tt.Elemwise)
        assert y.dtype == "bool"
        f = theano.function([x], y, allow_input_downcast=True)
        f([[0, 1, 2]])


class TestShape:
    def test_basic0(self):
        s = shape(np.ones((5, 3)))
        assert (eval_outputs([s]) == [5, 3]).all()

    def test_basic1(self):
        s = shape(np.ones(2))
        assert (eval_outputs([s]) == [2]).all()

    def test_basic2(self):
        s = shape(np.ones((5, 3, 10)))
        assert (eval_outputs([s]) == [5, 3, 10]).all()


class TestMaxAndArgmax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_basic(self):
        n = as_tensor_variable(5.0)
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 5.0
        assert i == 0
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0
        v = eval_outputs(max_and_argmax(n)[1].shape)
        assert len(v) == 0

    def test_basic_1(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 3
        assert i == 2
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0

    def test_basic_2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v_shape = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v_shape) == np.max(data, np_axis).shape

    def test_basic_2_float16(self):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (rand(20, 30).astype("float16") - 0.5) * 20
        n = shared(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis), (MaxAndArgmax,))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v_shape = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v_shape) == np.max(data, np_axis).shape

    def test_basic_2_invalid(self):
        n = as_tensor_variable(rand(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, 3))

        n = as_tensor_variable(rand(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, -3))

    def test_basic_2_valid_neg(self):
        n = as_tensor_variable(rand(2, 3))
        v, i = eval_outputs(max_and_argmax(n, -1))
        assert i.dtype == "int64"
        assert v.shape == (2,)
        assert i.shape == (2,)
        assert np.all(v == np.max(n.value, -1))
        assert np.all(i == np.argmax(n.value, -1))
        v, i = eval_outputs(max_and_argmax(n, -2))
        assert i.dtype == "int64"
        assert v.shape == (3,)
        assert i.shape == (3,)
        assert np.all(v == np.max(n.value, -2))
        assert np.all(i == np.argmax(n.value, -2))
        v = eval_outputs(max_and_argmax(n, -1)[0].shape)
        assert v == (2)
        v = eval_outputs(max_and_argmax(n, -2)[0].shape)
        assert v == (3)

    def test_basic_3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1, 2], None),
            ([1, 2, 0], None),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v) == np.max(data, np_axis).shape

    def test_arg_grad(self):
        # The test checks that the gradient of argmax(x).sum() is 0

        x = matrix()
        cost = argmax(x, axis=0).sum()
        gx = grad(cost, x)
        val = tt.get_scalar_constant_value(gx)
        assert val == 0.0

    def test_grad(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def safe_verify_grad(func, data):
            # Wrapper around 'verify_grad' that picks a proper value for epsilon.
            #
            # This is needed because 'verify_grad' may fail when its epsilon is
            # too large, due to the fact the argmax is not continuous.
            # We make sure epsilon is less than the minimum absolute value found
            # in the matrix of pairwise differences between all elements in the
            # data. This way, the argmax will not change when adding epsilon.

            # 'data' is a one-element list.
            (data_tensor,) = data
            # Flatten it into a 1D vector.
            data_vector = data_tensor.flatten()
            # Compute pairwise absolute differences.
            diff = np.abs(data_vector.reshape((-1, 1)) - data_vector)
            # Alter the diagonal to avoid a zero minimum.
            for i in range(len(diff)):
                diff[i, i] = 1
            # Find an appropriate epsilon.
            eps = builtins.min(numeric_grad.type_eps[config.floatX], diff.min() / 2)
            # Run gradient verification.
            utt.verify_grad(func, data, eps=eps)

        def check_grad_max(data, max_grad_data, axis=None):
            # Why this is needed? verify_grad is not enough?
            # This works only for axis in [0, None].
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[argmax] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        for axis in (-1, 0, 1, None):
            for j in range(2):
                safe_verify_grad(lambda v: max_and_argmax(v, axis=axis)[j], [data])
                if axis != 1:
                    safe_verify_grad(
                        lambda v: max_and_argmax(v.flatten(), axis=axis)[j], [data]
                    )
            if axis in (0, None):
                check_grad_max(
                    data,
                    eval_outputs(grad(max_and_argmax(n, axis=axis)[0].sum(), n)),
                    axis=axis,
                )
            check_grad_max(data, eval_outputs(grad(max_and_argmax(n.flatten())[0], n)))

        # Test 3d inner dimensions
        data = rand(3, 4, 5)

        for i in [0, 1, 2]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test 4d inner dimensions
        data = rand(2, 3, 4, 5)

        for i in [0, 1, 2, 3]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test grad with multiple axes
        for i in [[0, 1], [0, 0]]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[1], [data])

    def test_preserve_broadcastable(self):
        # Ensure the original broadcastable flags are preserved by Max/Argmax.
        x = tt.matrix().dimshuffle("x", 0, "x", 1, "x")
        y = x.max(axis=1)
        assert y.type.broadcastable == (True, True, False, True)

    def test_multiple_axes(self):
        data = np.arange(24).reshape(3, 2, 4)
        x = as_tensor_variable(data)
        v, i = eval_outputs(max_and_argmax(x, [1, -1]))
        assert np.all(v == np.array([7, 15, 23]))
        assert np.all(i == np.array([7, 7, 7]))

        v = eval_outputs(max_and_argmax(x, [1, -1])[0].shape)
        assert tuple(v) == np.max(data, (1, -1)).shape

    def test_zero_shape(self):
        x = tt.matrix()
        m, i = max_and_argmax(x, axis=1)
        f = theano.function([x], [m, i])
        xv = np.zeros((0, 4), dtype=config.floatX)
        mv, iv = f(xv)
        assert mv.shape == (0,)
        assert iv.shape == (0,)

    def test_numpy_input(self):
        ar = np.array([1, 2, 3])
        max, argmax = max_and_argmax(ar, axis=None)
        assert max.eval(), 3
        assert argmax.eval(), 2


class TestArgminArgmax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [argmin, argmax]:
            n = as_tensor_variable(5.0)
            i = eval_outputs(fct(n))
            assert i == 0
            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmin(n))
        assert i == 4
        v = eval_outputs(argmin(n).shape)
        assert len(v) == 0

        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmax(n))
        assert i == 2
        v = eval_outputs(argmax(n).shape)
        assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_float16(self):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (rand(20, 30).astype("float16") - 0.5) * 20
        n = shared(data)
        mode = get_default_mode().including("local_max_and_argmax", "uncanonicalize")
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis), (Argmax,), mode=mode)
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape, mode=mode)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            i = eval_outputs(fct(n, -1))
            assert i.shape == (2,)
            assert np.all(i == nfct(n.value, -1))
            i = eval_outputs(fct(n, -2))
            assert i.shape == (3,)
            assert np.all(i == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test_grad_argmin(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        n.name = "n"

        # test grad of argmin
        utt.verify_grad(lambda v: argmin(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmin(v.flatten()), [data])

        try:
            cost = argmin(n, axis=-1)
            cost.name = None
            grad(cost, n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_grad_argmax(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        # test grad of argmax
        utt.verify_grad(lambda v: argmax(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmax(v.flatten()), [data])

        try:
            grad(argmax(n, axis=-1), n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            i = eval_outputs(argmin(n))
            assert i == 1
            i = eval_outputs(argmax(n))
            assert i == 3

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        i = eval_outputs(argmin(n))
        assert i == 1
        i = eval_outputs(argmax(n))
        assert i == 0


class TestMinMax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [max, min]:
            n = as_tensor_variable(5.0)
            v = eval_outputs(fct(n))
            assert v == 5.0

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable([1, 2, 3, 2, -6])
            v = eval_outputs([fct(n)])
            assert v == nfct(n.value)

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct in [max, min]:
            n = as_tensor_variable(rand(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable(rand(2, 3))
            v = eval_outputs(fct(n, -1))
            assert v.shape == (2,)
            assert np.all(v == nfct(n.value, -1))
            v = eval_outputs(fct(n, -2))
            assert v.shape == (3,)
            assert np.all(v == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        # Test with 1 axis or all axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test3b(self):
        # Test with 2 axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for axis in [[0, 1], [1, 2], [0, 2]]:
                v = eval_outputs(fct(n, axis))
                np_v = nfct(nfct(data, axis[1]), axis[0])
                assert np.all(v == np_v)
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == np_v.shape

    def test_grad_max(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            # This work only for axis in [0,None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[np.argmax(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        # test grad of max
        # axis is the last one
        utt.verify_grad(lambda v: max(v, axis=-1), [data])

        utt.verify_grad(lambda v: max(v, axis=[0]), [data])
        check_grad_max(data, eval_outputs(grad(max(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: max(v, axis=[1]), [data])
        # check_grad_max(data,eval_outputs(grad(max(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: max(v.flatten()), [data])
        check_grad_max(data, eval_outputs(grad(max(n.flatten()), n)))

    def test_grad_min(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_min(data, min_grad_data, axis=None):
            # This work only for axis in [0, None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmin = np.argmin(data, axis=axis)
            if argmin.ndim == 0:
                z[np.argmin(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmin):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(min_grad_data == z)

        # test grad of min
        # axis is the last one
        utt.verify_grad(lambda v: min(v, axis=-1), [data])

        utt.verify_grad(lambda v: min(v, axis=[0]), [data])
        check_grad_min(data, eval_outputs(grad(min(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: min(v, axis=[1]), [data])
        # check_grad_min(data,eval_outputs(grad(min(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: min(v.flatten()), [data])
        check_grad_min(data, eval_outputs(grad(min(n.flatten()), n)))

    def _grad_list(self):
        # Test the gradient when we have multiple axis at the same time.
        #
        # This not implemented, so we disable the test. See ticket:
        # http://www.assembla.com/spaces/theano/tickets/511
        data = rand(2, 3)
        for fct in [max_and_argmax, max, min]:
            utt.verify_grad(lambda v: fct(v, axis=[0, 1]), [data])
        # n = as_tensor_variable(data)
        # check_grad_max(data, eval_outputs(grad(max_and_argmax(n,
        # axis=1)[0], n)),axis=1)

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            assert min(n).dtype == dtype
            i = eval_outputs(min(n))
            assert i == itype.min
            assert max(n).dtype == dtype
            i = eval_outputs(max(n))
            assert i == itype.max

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        assert min(n).dtype == "bool"
        i = eval_outputs(min(n))
        assert i.ndim == 0
        assert not np.any(i)
        assert max(n).dtype == "bool"
        i = eval_outputs(max(n))
        assert i.ndim == 0
        assert np.all(i)


def test_basic_allclose():
    # This was raised by a user in https://github.com/Theano/Theano/issues/2975
    assert tt.basic._allclose(-0.311023883434, -0.311022856884)


class TestOuter:
    def test_outer(self):
        for m in range(4):
            for n in range(4):
                x = tt.tensor(dtype="floatX", broadcastable=(False,) * m)
                y = tt.tensor(dtype="floatX", broadcastable=(False,) * n)
                s1 = np.random.randint(1, 10, m)
                s2 = np.random.randint(1, 10, n)
                v1 = np.asarray(np.random.rand(*s1)).astype(config.floatX)
                v2 = np.asarray(np.random.rand(*s2)).astype(config.floatX)
                o = tt.outer(x, y).eval({x: v1, y: v2})
                assert_allclose(o, np.outer(v1, v2))

    def test_grad(self):
        # Test the combined graph of the graph of outer
        # with broadcastable dimensions, just in case.
        for shp0, shp1 in [
            ((1,), (2,)),
            ((3,), (1,)),
            ((1,), (1,)),
            ((3,), (2,)),
            ((3, 2), (1, 1)),
            ((3, 2), (1, 4)),
            ((3, 2), (4, 1)),
            ((3, 2), (4, 5)),
            ((1, 2), (4, 5)),
            ((3, 1), (4, 5)),
            ((1, 1), (4, 5)),
            ((1, 1), (1, 1)),
        ]:
            data0 = np.random.rand(*shp0).astype(config.floatX)
            data1 = np.random.rand(*shp1).astype(config.floatX)
            utt.verify_grad(tt.outer, [data0, data1])


class TestGetVectorLength:
    def test_get_vector_length(self):
        x = theano.shared(np.zeros((2, 3, 4, 5)))
        assert len(list(x.shape)) == 4
        assert len(list(x.shape[2:4])) == 2
        assert len(list(x.shape[2:])) == 2
        assert len(list(x.shape[1:4])) == 3
        assert len(list(x.shape[2:2])) == 0
        assert len(list(x.shape[1:5])) == 3
        assert len(list(x.shape[1:10])) == 3
        # Test step
        assert len(list(x.shape[1:10:2])) == 2
        # Test neg start
        assert len(list(x.shape[-1:4])) == 1
        assert len(list(x.shape[-6:4])) == 4
        # test neg stop
        assert len(list(x.shape[1:-2])) == 1
        assert len(list(x.shape[1:-1])) == 2


class TestJoinAndSplit:
    # Split is tested by each verify_grad method.
    def setup_method(self):
        Join.debug = False
        utt.seed_rng()
        self.mode = theano.compile.get_default_mode().excluding("constant_folding")
        self.join_op = Join()
        self.split_op_class = Split
        self.make_vector_op = opt.MakeVector()
        self.floatX = config.floatX
        self.hide_error = theano.config.mode not in [
            "DebugMode",
            "DEBUG_MODE",
            "FAST_COMPILE",
        ]
        self.shared = shared

    def eval_outputs_and_check_join(self, outputs):
        f = theano.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def eval_outputs_and_check_vector(self, outputs, make_vector_op=None):
        if make_vector_op is None:
            make_vector_op = self.make_vector_op
        f = theano.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(make_vector_op))]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def test_join_scalar(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        with pytest.raises(TypeError):
            join(0, a, b)

    def test_stack_mixed_type_constants(self):
        # tested only on cpu as gpu support only float32
        a = as_tensor_variable(1)
        b = as_tensor_variable(2.0)
        c = tt._shared(np.asarray(3.0, dtype=self.floatX))
        s = stack([a, b, c])
        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s], opt.MakeVector())
        assert (out == want).all()

    def test_stack_scalar(self):
        a = self.shared(np.asarray(1.0, dtype=self.floatX))
        b = as_tensor_variable(2.0)
        c = as_tensor_variable(3.0)
        s = stack([a, b, c])

        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s])
        assert (out == want).all()

    def test_stack_scalar_make_vector(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # not Join. Test that the floatX dtype stay floatX, not downcasted
        # to int64
        a = tt.scalar("a", dtype=self.floatX)
        b = tt.scalar("b", dtype=self.floatX)
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        # print val
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == self.floatX

    def test_stack_scalar_make_vector_dtype(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # event when the scalar don't have the same dtype.
        a = tt.iscalar("a")
        b = tt.lscalar("b")
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_scalar_make_vector_constant(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # event when the scalar are simple int type.
        a = tt.iscalar("a")
        b = tt.lscalar("b")
        # test when the constant is the first element.
        # The first element is used in a special way
        s = stack([10, a, b, np.int8(3)])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [10, 1, 2, 3])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_new_interface(self):
        # Test the new numpy-like interface: stack(tensors, axis=0).

        # Testing against old interface
        warnings.simplefilter("always", DeprecationWarning)
        a = tt.imatrix("a")
        b = tt.imatrix("b")
        s1 = stack(a, b)
        s2 = stack([a, b])
        f = function([a, b], [s1, s2], mode=self.mode)
        v1, v2 = f([[1, 2]], [[3, 4]])
        assert v1.shape == v2.shape
        assert np.all(v1 == v2)
        # Testing axis parameter
        s3 = stack([a, b], 1)
        f = function([a, b], s3, mode=self.mode)
        v3 = f([[1, 2]], [[3, 4]])
        v4 = np.array([[[1, 2], [3, 4]]])
        assert v3.shape == v4.shape
        assert np.all(v3 == v4)
        # Testing negative axis
        v1 = [[1, 2, 3], [4, 5, 6]]
        v2 = [[7, 8, 9], [10, 11, 12]]
        s = stack([a, b], axis=-1)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 3, 2))
        v[:, :, 0] = v1
        v[:, :, 1] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        s = stack([a, b], axis=-2)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 2, 3))
        v[:, 0, :] = v1
        v[:, 1, :] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        # Testing out-of-bounds axis
        with pytest.raises(IndexError):
            stack([a, b], 4)
        with pytest.raises(IndexError):
            stack([a, b], -4)
        # Testing depreciation warning
        with warnings.catch_warnings(record=True) as w:
            s = stack(a, b)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
        with warnings.catch_warnings(record=True) as w:
            s = stack([a, b])
            s = stack([a, b], 1)
            s = stack([a, b], axis=1)
            s = stack(tensors=[a, b])
            s = stack(tensors=[a, b], axis=1)
            assert not w

    def test_stack_hessian(self):
        # Test the gradient of stack when used in hessian, see gh-1589
        a = tt.dvector("a")
        b = tt.dvector("b")
        A = stack([a, b])
        B = A.T.dot(A)
        Ha, Hb = hessian(B.sum(), [a, b])

        # Try some values
        a_v = np.random.rand(4)
        b_v = np.random.rand(4)
        f = theano.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 2
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 2.0)
        assert np.allclose(Hb_v, 2.0)

    def test_stack_hessian2(self):
        # Test the hessian macro when the gradient itself does not depend
        # on the input (but the cost does)
        a = tt.dvector("a")
        b = tt.dvector("b")
        A = stack([a, b])
        Ha, Hb = hessian(A.sum(), [a, b])

        # Try some values
        a_v = np.random.rand(4)
        b_v = np.random.rand(4)
        f = theano.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 0
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 0.0)
        assert np.allclose(Hb_v, 0.0)

    def test_join_concatenate_one_element(self):
        # Fast test of concatenate as this is an alias for join.
        # also test that we remove the Join op if there is only 1 input
        m = tt.fmatrix()
        c = tt.concatenate([m])
        f = theano.function(
            inputs=[m], outputs=[c], mode=self.mode.including("local_join_1")
        )
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

    def test_join_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = join(0, a, b)
        want = np.array([1, 2, 3, 7, 8, 9])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_roll(self):

        for get_shift in [lambda a: a, lambda x: theano.shared(x)]:
            # Test simple 1D example
            a = self.shared(np.array([1, 2, 3, 4, 5, 6], dtype=self.floatX))
            b = roll(a, get_shift(2))
            want = np.array([5, 6, 1, 2, 3, 4])
            out = theano.function([], b)()

            assert (out == want).all()

            # Test simple 1D example with explicit 0 axis
            b = roll(a, get_shift(-1), 0)
            want = np.array([2, 3, 4, 5, 6, 1])
            out = theano.function([], b)()

            assert (out == want).all()

            # Test 2D example - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(21).reshape((3, 7)).astype(self.floatX))
            b = roll(a, get_shift(-2), 1)

            want = np.roll(a.get_value(borrow=True), -2, 1)
            out = theano.function([], b)()

            assert (out == want).all()

            # Test example when axis < 0 - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(24).reshape((3, 2, 4)).astype(self.floatX))
            b = roll(a, get_shift(-2), -2)

            want = np.roll(a.get_value(borrow=True), -2, -2)
            out = theano.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0
            want = np.roll(a.get_value(borrow=True), -2, 0)
            b = roll(a, get_shift(-2), 0)
            out = theano.function([], b)()

            assert (out == want).all()

            # Test rolling on default axis with ndim > 1
            want = np.roll(a.get_value(borrow=True), 2)
            b = roll(a, get_shift(2))
            out = theano.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a positive shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), 4, 0)
            b = roll(a, get_shift(4), 0)
            out = theano.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a negative shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), -4, 0)
            b = roll(a, get_shift(-4), 0)
            out = theano.function([], b)()

            assert (out == want).all()

    def test_stack_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = stack([a, b])
        want = np.array([[1, 2, 3], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix0(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        s = join(0, a, b)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix1(self):
        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[0.1, 0.2, 0.3, 0.7], [0.4, 0.5, 0.6, 0.8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrix_dtypes(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        grad(s.sum(), b)
        grad(s.sum(), a)
        utt.verify_grad(lambda b: join(1, a, b), [bv], eps=1.0e-2, mode=self.mode)

    def test_join_matrix_ints(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="int32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        assert (np.asarray(grad(s.sum(), b).eval()) == 0).all()
        assert (np.asarray(grad(s.sum(), a).eval()) == 0).all()

    def test_join_matrix1_using_vertical_stack(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        c = as_tensor_variable(np.array([[9, 8, 7]], dtype=self.floatX))
        s = vertical_stack(a, b, c)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix1_using_horizontal_stack(self):
        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        cv = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        c = as_tensor_variable(cv)
        s = horizontal_stack(a, b, c)
        want = np.array(
            [[0.1, 0.2, 0.3, 0.7, 0.3, 0.2, 0.1], [0.4, 0.5, 0.6, 0.8, 0.6, 0.5, 0.4]],
            dtype="float32",
        )
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrixV(self):
        # variable join axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(0)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )
        got = f(1)
        assert np.allclose(got, want)

        utt.verify_grad(lambda a, b: join(0, a, b), [v, 2 * v], mode=self.mode)
        utt.verify_grad(lambda a, b: join(1, a, b), [v, 2 * v], mode=self.mode)

    def test_join_matrixV_negative_axis(self):
        # variable join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f(-1)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(-2)
        assert np.allclose(got, want)

        with pytest.raises(IndexError):
            f(-3)

    def test_join_matrixC_negative_axis(self):
        # constant join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)

        s = join(-1, a, b)
        f = theano.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        s = join(-2, a, b)
        f = theano.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        with pytest.raises(IndexError):
            join(-3, a, b)

        utt.verify_grad(lambda a, b: join(-1, a, b), [v, 2 * v], mode=self.mode)

    def test_vector_len(self):
        x = lscalar("x")
        y = dscalar("y")

        triple = as_tensor_variable((x, y, 9.0))
        assert 3 == get_vector_length(triple)

        a, b, c = triple
        f = function([x, y], [b, c, a], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, opt.MakeVector)]

        assert np.allclose(f(4, 5), [5, 9, 4])

    def test_broadcastable_flag_assignment_mixed_otheraxes(self):
        # Test that the broadcastable flags for the output of
        # a join operation on non-join axes are True if one or
        # more inputs is broadcastable on that dimension.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 3, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(1, a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # Opt can remplace the int by a Theano constant
        c = self.join_op(tt.constant(1), a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # In case futur opt insert other useless stuff
        c = self.join_op(tt.cast(tt.constant(1), dtype="int32"), a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(1, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

        # Should raise an error if dimension 0 does not match
        a.set_value(rng.rand(2, 4, 1).astype(self.floatX))
        with pytest.raises(ValueError):
            f()

    def test_broadcastable_flag_assignment_mixed_thisaxes(self):
        # Test that the broadcastable flag of the join axis
        # is False when some inputs are broadcastable on that
        # dimension.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(2, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 4, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )
        # Should raise an error if b_val.shape[0] is not 1
        # We can't set the value|
        with pytest.raises(TypeError):
            b.set_value(rng.rand(3, 4, 1).astype(self.floatX))
        a = TensorType(dtype=self.floatX, broadcastable=[0, 0, 1])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1])()
        c = self.join_op(0, a, b)
        f = function([a, b], c, mode=self.mode)
        bad_b_val = rng.rand(3, 4, 1).astype(self.floatX)
        with pytest.raises(TypeError):
            f(a_val, bad_b_val)

    def test_broadcastable_flags_all_broadcastable_on_joinaxis(self):
        # Test that joining together several inputs which are all
        # broadcastable on the join dimension results in the output
        # being non-broadcastable on the join dimension.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 4, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

    def test_broadcastable_single_input_broadcastable_dimension(self):
        # Test that all broadcastable flags are preserved by a
        # single-input join.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.join_op(0, a)
        assert b.type.broadcastable[0]
        assert b.type.broadcastable[2]
        assert not b.type.broadcastable[1]

        f = function([], b, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        if theano.config.mode != "FAST_COMPILE":
            assert not [
                True for node in topo if isinstance(node.op, type(self.join_op))
            ]

        f()
        utt.verify_grad((lambda a: join(0, a)), [a_val], rng=rng, mode=self.mode)
        # Should raise an error if length of dimension 0 is not 1
        with pytest.raises(TypeError):
            a.set_value(rng.rand(2, 4, 1).astype(self.floatX))
        # with pytest.raises(TypeError):
        #    f(bad_a_val)

    def test_broadcastable_flags_many_dims_and_inputs(self):
        # Test that the right broadcastable flags get set for a join
        # with many inputs and many input dimensions.
        a = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 0])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 1, 1, 0, 0, 0])()
        c = TensorType(dtype=self.floatX, broadcastable=[1, 0, 0, 0, 0, 0])()
        d = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 1, 0, 1])()
        e = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 1])()
        f = self.join_op(0, a, b, c, d, e)
        fb = f.type.broadcastable
        assert not fb[0] and fb[1] and fb[2] and fb[3] and not fb[4] and fb[5]
        g = self.join_op(1, a, b, c, d, e)
        gb = g.type.broadcastable
        assert gb[0] and not gb[1] and gb[2] and gb[3] and not gb[4] and gb[5]
        h = self.join_op(4, a, b, c, d, e)
        hb = h.type.broadcastable
        assert hb[0] and hb[1] and hb[2] and hb[3] and not hb[4] and hb[5]

        f = function([a, b, c, d, e], f, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        b_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        c_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        d_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        e_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        f(a_val, b_val, c_val, d_val, e_val)
        utt.verify_grad(
            (lambda a, b, c, d, e: join(0, a, b, c, d, e)),
            [a_val, b_val, c_val, d_val, e_val],
            rng=rng,
            mode=self.mode,
        )
        # Should raise an error if length of dimension 0 is not 1
        bad_val = rng.rand(2, 1, 1, 1, 2, 1).astype(self.floatX)
        with pytest.raises(TypeError):
            f(bad_val, b_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, bad_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, bad_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, bad_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, d_val, bad_val)
        # Should raise an error if any dimension other than 4 has length != 1
        bad_a_val = rng.rand(1, 2, 1, 1, 2, 1).astype(self.floatX)
        bad_b_val = rng.rand(1, 1, 1, 1, 2, 2).astype(self.floatX)
        bad_c_val = rng.rand(1, 1, 2, 1, 2, 1).astype(self.floatX)
        bad_d_val = rng.rand(1, 2, 1, 1, 2, 1).astype(self.floatX)
        bad_e_val = rng.rand(1, 1, 1, 2, 2, 1).astype(self.floatX)
        with pytest.raises(ValueError):
            f(bad_a_val, b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, bad_b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, bad_c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, bad_d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, d_val, bad_e_val)

    def test_infer_shape_join(self):
        def get_mat(s1, s2):
            return np.asarray(np.random.uniform(size=(s1, s2)), dtype=self.floatX)

        x1 = self.shared(get_mat(3, 4))
        x2 = self.shared(get_mat(2, 4))
        x3 = self.shared(get_mat(1, 4))

        # Test dim 0
        z = self.join_op(0, x1, x2, x3)
        f = theano.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()

        out = f()
        assert (out == [6, 4]).all()

        if theano.config.mode != "FAST_COMPILE":
            for node in f.maker.fgraph.toposort():
                assert not isinstance(node.op, type(self.join_op))

        # Test dim 1
        x1.set_value(get_mat(3, 4))
        x2.set_value(get_mat(3, 4))
        x3.set_value(get_mat(3, 5))
        z = self.join_op(1, x1, x2, x3)
        f = theano.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        out = f()
        assert (out == [3, 13]).all()

        if theano.config.mode != "FAST_COMPILE":
            for node in topo:
                assert not isinstance(node.op, type(self.join_op))

        with change_flags(compute_test_value="off"):
            # Test hide error
            x1.set_value(get_mat(3, 4))
            x2.set_value(get_mat(3, 4))
            x3.set_value(get_mat(2, 5))
            if not self.hide_error:
                with pytest.raises(ValueError):
                    f()
            else:
                f()

    def test_rebroadcast(self):
        # Regression test for a crash that used to happen when rebroadcasting.
        x = tt.TensorType(self.floatX, [False, False, True])()
        u = tt.TensorType(self.floatX, [False, False, True])()
        # This line used to crash.
        tt.concatenate([x, -u], axis=2)

    def test_concatenate_same(self):
        # Test that we can concatenate the same tensor multiple time.

        # In the past it was broken on the GPU.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        T_shared = self.shared(rng.rand(3, 4).astype(self.floatX))
        Tout = tt.concatenate([T_shared, T_shared])
        f = function([], Tout, mode=self.mode)
        out = f()
        if theano.config.mode != "FAST_COMPILE":
            assert [
                True
                for node in f.maker.fgraph.toposort()
                if isinstance(node.op, type(self.join_op))
            ]
        assert np.allclose(
            out, np.concatenate([T_shared.get_value(), T_shared.get_value()])
        )

    def test_mixed_ndim_error(self):
        rng = np.random.RandomState(seed=utt.fetch_seed())
        v = self.shared(rng.rand(4).astype(self.floatX))
        m = self.shared(rng.rand(4, 4).astype(self.floatX))
        with pytest.raises(TypeError):
            self.join_op(0, v, m)

    def test_split_0elem(self):
        rng = np.random.RandomState(seed=utt.fetch_seed())
        m = self.shared(rng.rand(4, 6).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [4, 0])
        f = function([], o, mode=self.mode)
        assert any(
            [
                isinstance(node.op, self.split_op_class)
                for node in f.maker.fgraph.toposort()
            ]
        )
        o1, o2 = f()
        assert np.allclose(o1, m.get_value(borrow=True))
        assert np.allclose(o2, m.get_value(borrow=True)[4:])

    @change_flags(compute_test_value="off")
    def test_split_neg(self):
        rng = np.random.RandomState(seed=utt.fetch_seed())
        m = self.shared(rng.rand(4, 6).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [5, -1])
        f = function([], o, mode=self.mode)
        assert any(
            [
                isinstance(node.op, self.split_op_class)
                for node in f.maker.fgraph.toposort()
            ]
        )
        with pytest.raises(ValueError):
            f()


def test_join_inplace():
    # Test join to work inplace.
    #
    # This function tests the case when several elements are passed to the
    # join function but all except one of them are empty. In this case join
    # should work inplace and the output should be the view of the non-empty
    # element.
    s = tt.lscalar()
    x = tt.vector("x")
    z = tt.zeros((s,))

    join = Join(view=0)
    c = join(0, x, z, z)

    f = theano.function([theano.In(x, borrow=True), s], theano.Out(c, borrow=True))

    data = np.array([3, 4, 5], dtype=theano.config.floatX)
    print(f(data, 0))

    if theano.config.mode not in ["DebugMode", "DEBUG_MODE"]:
        assert f(data, 0) is data
    assert np.allclose(f(data, 0), [3, 4, 5])


def test_join_oneInput():
    # Test join when only 1 input is given.
    #
    # This functions tests the case when concatenate is called
    # on an array of tensors but the array has only one element.
    # In this case, we would like to avoid the computational
    # overhead of concatenation of one element.
    x_0 = tt.fmatrix()
    x_1 = tt.fmatrix()
    x_2 = tt.fvector()
    join_0 = tt.concatenate([x_0], axis=1)
    join_1 = tt.concatenate([x_0, x_1, tt.shape_padright(x_2)], axis=1)

    assert join_0 is x_0
    assert join_1 is not x_0


class TestComparison:
    # Test <, >, <=, >=, == and !=
    #
    # Test that we can do the comparison with different
    # combination of tensor(shared and constant variable) with
    # ndarray. ndarray cmp tensor was crashing.  In a NumPy PR (should
    # be in the release 1.8 of NumPy), it will work.  So we assert it
    # work(futur behavior) or raise an error(current NumPy release).
    def setup_method(self):
        utt.seed_rng()
        self.mode = None
        self.shared = shared
        self.dtypes = ["float64", "float32", "complex64", "complex128"]

    def inplace_func(self, inputs, outputs, check_isfinite=None):
        mode = self.mode
        if check_isfinite is False:
            if mode is None:
                mode = get_default_mode()
            mode.check_isfinite = False
        f = inplace_func(inputs, outputs, mode=mode)
        return f

    def test_gt(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x > y)
                    v = fn()
                    assert np.all(v == (l > r)), (v, (l > r))
                except TypeError:
                    assert err

    def test_lt(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x < y)
                    v = fn()
                    assert np.all(v == (l < r)), (v, (l < r))
                except TypeError:
                    assert err

    def test_le(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x <= y)
                    v = fn()
                    assert np.all(v == (l <= r)), (v, (l <= r))
                except TypeError:
                    assert err

    def test_ge(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x >= y)
                    v = fn()
                    assert np.all(v == (l >= r)), (v, (l >= r))
                except TypeError:
                    assert err

    def test_eq(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], eq(x, y))
                    v = fn()
                    assert np.all(v == (l == r)), (v, (l == r))
                except TypeError:
                    assert err

    def test_neq(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (tt.constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), tt.constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], neq(x, y))
                    v = fn()
                    assert np.all(v == (l != r)), (v, (l != r))
                except TypeError:
                    assert err

    def test_isclose(self):
        for dtype in self.dtypes:
            l = np.asarray(
                [0.0, 1.0, -1.0, 0.0, np.nan, np.inf, -np.inf, np.inf], dtype=dtype
            )
            r = np.asarray(
                [0.0, 1.0001, -1.000000000001, np.nan, np.nan, np.inf, np.inf, 0.0],
                dtype=dtype,
            )
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    o1 = isclose(x, y, equal_nan=False)
                    fn1 = self.inplace_func([], o1, check_isfinite=False)

                    o2 = isclose(x, y, equal_nan=True)
                    fn2 = self.inplace_func([], o2, check_isfinite=False)

                    v1 = fn1()
                    v2 = fn2()
                    assert np.all(
                        v1
                        == np.asarray(
                            [True, False, True, False, False, True, False, False],
                            dtype="bool",
                        )
                    )
                    assert np.all(
                        v2
                        == np.asarray(
                            [True, False, True, False, True, True, False, False],
                            dtype="bool",
                        )
                    )
                except TypeError:
                    if not dtype.startswith("complex"):
                        raise
                        assert err

    def test_allclose(self):
        # equal_nan argument not in current version of numpy allclose,
        # force it to False.
        for dtype in self.dtypes:
            l = np.asarray(
                [0.0, 1.0, -1.0, 0.0, np.nan, np.inf, -np.inf, np.inf], dtype=dtype
            )
            r = np.asarray(
                [0.0, 1.0001, -1.000000000001, np.nan, np.nan, np.inf, np.inf, 0.0],
                dtype=dtype,
            )
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func(
                        [], allclose(x, y, equal_nan=False), check_isfinite=False
                    )
                    v = fn()
                    assert np.all(v == np.allclose(l, r))
                except TypeError:
                    if not dtype.startswith("complex"):
                        assert err


class TestBitwise:
    dtype = [
        "int8",
        "int16",
        "int32",
        "int64",
    ]

    def test_or(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x | y)
            l = theano._asarray([0, 0, 1, 1], dtype=dtype)
            r = theano._asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.or_(l, r))), (l, r, v)

    def test_XOR(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x ^ y)
            l = theano._asarray([0, 0, 1, 1], dtype=dtype)
            r = theano._asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.xor(l, r))), (l, r, v)

    def test_and(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x & y)
            l = theano._asarray([0, 0, 1, 1], dtype=dtype)
            r = theano._asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.and_(l, r))), (l, r, v)

    def test_inv(self):
        for dtype in self.dtype:
            x = vector(dtype=dtype)
            fn = inplace_func([x], ~x)
            for l in [
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [-1, 2 ** 16, 2 ** 16 - 1],
            ]:
                l = theano._asarray([0, 0, 1, 1], dtype=dtype)
                v = fn(l)
                assert np.all(v == (~l)), (l, v)

    def test_eye(self):
        n = iscalar()
        m = iscalar()
        k = iscalar()
        fn = theano.function([m, n, k], eye(m, n, k))
        assert np.all(fn(5, 6, 1) == np.eye(5, 6, 1))


class TestAdd:
    def setup_method(self):
        utt.seed_rng()

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = shared(np.ones(3, dtype="complex%i" % nbits) + 0.5j)
            b = shared(np.ones(3, dtype="complex%i" % nbits) + 1.5j)
            tests = (
                ("+", lambda x, y: x + y),
                ("-", lambda x, y: x - y),
                ("*", lambda x, y: x * y),
                ("/", lambda x, y: x / y),
            )
            for s, fn in tests:
                f = inplace_func([], fn(a, b))
                # print 'valid output:', fn(a.data, b.data)
                # print 'theano output:', f(a.data, b.data)
                assert a.type.values_eq_approx(fn(a.get_value(), b.get_value()), f())

    def test_grad_scalar_l(self):
        utt.verify_grad(add, [np.asarray([3.0]), rand(3)])

    def test_grad_scalar_r(self):
        utt.verify_grad(add, [rand(3), np.asarray([3.0])])

    def test_grad_row(self):
        utt.verify_grad(add, [rand(3, 5), rand(1, 5)])

    def test_grad_col(self):
        utt.verify_grad(add, [rand(3, 5), rand(3, 1)])


class TestCeil:
    def test_complex(self):
        with pytest.raises(TypeError):
            tt.ceil(tt.zvector())


class TestExp:
    def test_grad_0(self):
        utt.verify_grad(
            exp,
            [
                np.asarray(
                    [
                        [1.5089518, 1.48439076, -4.7820262],
                        [2.04832468, 0.50791564, -1.58892269],
                    ]
                )
            ],
        )

    def test_int(self):
        x = ivector()
        f = function([x], exp(x))
        exp_3 = f([3])
        assert exp_3.dtype == "float64"

    def test_complex(self):
        x = zvector()
        assert exp(x).dtype == "complex128"
        f = function([x], exp(x))
        exp_3 = f([3 + 2j])
        assert np.allclose(exp_3, np.exp(3 + 2j))


class TestDivimpl:
    def test_impls(self):
        i = iscalar()
        ii = lscalar()
        d = dscalar()
        f = fscalar()
        c = cscalar()

        assert np.allclose(function([i, d], i / d)(5, 7.0), (5.0 / 7.0))
        assert np.allclose(function([i, d], d / i)(5, 7.0), (7.0 / 5.0))
        assert np.allclose(function([i, f], i / f)(5, 11.0), (5.0 / 11.0))
        assert np.allclose(function([i, f], f / i)(5, 11.0), (11.0 / 5.0))
        assert np.allclose(function([i, ii], i // ii)(5, 3), (5 // 3))
        assert np.allclose(function([i, ii], ii // i)(5, 3), (3 // 5))
        assert np.allclose(function([i, ii], true_div(i, ii))(5, 3), (5.0 / 3.0))
        assert np.allclose(function([i, ii], true_div(ii, i))(5, 3), (3.0 / 5.0))
        assert np.allclose(
            function([i, c], i / c)(5, np.complex(5, 3)), (5.0 / (5 + 3j))
        )
        assert np.allclose(
            function([i, c], c / i)(5, np.complex(5, 3)), ((5 + 3j) / 5.0)
        )


class TestMean:
    def test_mean_single_element(self):
        res = tt.mean(np.zeros(1))
        assert res.eval() == 0.0

    def test_mean_f16(self):
        x = tt.vector(dtype="float16")
        y = x.mean()
        f = theano.function([x], y)
        utt.assert_allclose(f(np.ones((100000,), dtype="float16")), 1.0)

    def test_basic(self):
        x = tt.vector()
        f = theano.function([x], tt.mean(x))
        data = rand(50)
        assert np.allclose(f(data), np.mean(data))

    def test_list(self):
        ll = [theano.shared(0.0), theano.shared(2.0)]
        assert tt.mean(ll).eval() == 1


class TestMatinv:
    def mat_reciprocal(self, dim):
        # symbolic program
        # broadcastable=[False,False] means that the shape of matrix is two dimensional,
        # and none of the dimensions are constrained to have length 1.
        # Note that TensorType's constructor does not actually allocate any memory.
        # TODO: Make TensorType syntax more explicit, and maybe give shape or number of dimensions.

        rng = np.random.RandomState(seed=utt.fetch_seed())

        a, b = matrices("ab")
        ab = a * b
        # Here, as_tensor_variable actually uses the data allocated by np.
        diff = ab - as_tensor_variable(np.ones((dim, dim), dtype=config.floatX))
        # Sum of squared errors
        ssdiff = sum(diff ** 2.0)

        g_b = grad(ssdiff, b)

        # compilation to function
        # [a,b] are the inputs, [ssdiff,g_b] are the outputs
        fn = inplace_func([a, b], [ssdiff, g_b])

        # use the function
        x = rng.rand(dim, dim) + 0.1  # Initialized s.t. x is not too tiny
        w = rng.rand(dim, dim)
        x = np.asarray(x, dtype=config.floatX)
        w = np.asarray(w, dtype=config.floatX)

        for i in range(100):
            ssd, gw = fn(x, w)
            # print ssd, x*w, x, w
            if i == 0:
                ssd0 = ssd
            w -= 0.4 * gw

        return ssd0, ssd

    def test_reciprocal(self):
        # Matrix reciprocal by gradient descent
        ssd0, ssd = self.mat_reciprocal(3)

        rng = np.random.RandomState(seed=utt.fetch_seed())
        # hand-coded numpy implementation for verification
        x = rng.rand(3, 3) + 0.1
        w = rng.rand(3, 3)
        x = np.asarray(x, dtype=config.floatX)
        w = np.asarray(w, dtype=config.floatX)
        ones = np.ones((3, 3), dtype=config.floatX)

        myssd0 = np.sum((x * w - ones) ** 2.0)
        # we want at least a test that is not too fast. So we make one here.
        for i in range(100):
            gw = 2 * (x * w - ones) * x  # derivative of dMSE/dw
            myssd = np.sum((x * w - ones) ** 2)
            w -= 0.4 * gw
        assert_almost_equal(ssd0, myssd0)
        assert_almost_equal(ssd, myssd)


def test_dot_numpy_inputs():
    """Test the `theano.tensor.dot` interface function with NumPy inputs."""
    a = np.ones(2)
    b = np.ones(2)
    res = tt.dot(a, b)
    assert isinstance(res, Variable)
    assert isinstance(res.owner.op, Dot)


class TestDot:
    def setup_method(self):
        utt.seed_rng()

    def test_Op_dims(self):
        _dot = tt.basic._dot
        d0 = scalar()
        d1 = vector()
        d2 = matrix()
        d3 = tensor3()

        with pytest.raises(TypeError):
            _dot(d0, d0)
        with pytest.raises(TypeError):
            _dot(d0, d1)
        with pytest.raises(TypeError):
            _dot(d0, d2)
        with pytest.raises(TypeError):
            _dot(d0, d3)
        with pytest.raises(TypeError):
            _dot(d1, d0)
        _dot(d1, d1)
        _dot(d1, d2)
        with pytest.raises(TypeError):
            _dot(d1, d3)
        with pytest.raises(TypeError):
            _dot(d2, d0)
        _dot(d2, d1)
        _dot(d2, d2)
        with pytest.raises(TypeError):
            _dot(d2, d3)
        with pytest.raises(TypeError):
            _dot(d3, d0)
        with pytest.raises(TypeError):
            _dot(d3, d1)
        with pytest.raises(TypeError):
            _dot(d3, d2)
        with pytest.raises(TypeError):
            _dot(d3, d3)

    def test_grad(self):
        utt.verify_grad(dense_dot, [rand(2, 3), rand(3, 2)])
        utt.verify_grad(dense_dot, [rand(2), rand(2, 3)])
        utt.verify_grad(dense_dot, [rand(3, 2), rand(2)])
        utt.verify_grad(dense_dot, [rand(2), rand(2)])
        utt.verify_grad(dense_dot, [rand(), rand()])
        # TODO: What about the broadcastable conditions in `Dot.grad`?

    def test_broadcastable_patterns(self):

        #
        # These examples should all work.  All dimensions of all results have
        # size 1.
        #
        def val_for(r):
            if r.dtype.startswith("complex"):
                # We want to test complex at the same time, so we give a value
                # to the imaginary component.
                # This strange way of doing things is the only way that worked
                # on NumPy 1.4.1.
                if r.ndim == 0:
                    return np.asarray(np.complex(1.1, 2.1), dtype=r.dtype)
                if r.ndim == 1:
                    if r.dtype == "complex64":
                        return np.complex64([np.complex(1.2, 2.2)])
                    elif r.dtype == "complex128":
                        return np.complex128([np.complex(1.2, 2.2)])
                elif r.ndim == 2:
                    if r.dtype == "complex64":
                        return np.complex64([[np.complex(1.3, 2.3)]])
                    elif r.dtype == "complex128":
                        return np.complex128([[np.complex(1.3, 2.3)]])

            if r.ndim == 0:
                return np.asarray(1.1, dtype=r.dtype)
            if r.ndim == 1:
                return np.asarray([1.2], dtype=r.dtype)
            elif r.ndim == 2:
                return np.asarray([[1.3]], dtype=r.dtype)
            raise AssertionError()

        for dtype0 in ("float32", "float64", "complex64"):
            for dtype1 in ("float32", "complex64", "complex128"):
                for bc0 in (
                    (True,),
                    (False,),
                    (True, True),
                    (True, False),
                    (False, True),
                    (False, False),
                ):
                    x = TensorType(dtype=dtype0, broadcastable=bc0)()
                    for bc1 in (
                        (True,),
                        (False,),
                        (True, True),
                        (True, False),
                        (False, True),
                        (False, False),
                    ):

                        y = TensorType(dtype=dtype1, broadcastable=bc1)()
                        z = dense_dot(x, y)

                        if dtype0.startswith("float") and dtype1.startswith("float"):
                            g = grad(z.sum(), x)
                            assert g.broadcastable == x.broadcastable
                            g = grad(z.sum(), y)
                            assert g.broadcastable == y.broadcastable


def test_TensorFromScalar():
    s = scal.constant(56)
    t = tensor_from_scalar(s)
    assert t.owner.op is tensor_from_scalar
    assert t.type.broadcastable == (), t.type.broadcastable
    assert t.type.ndim == 0, t.type.ndim
    assert t.type.dtype == s.type.dtype

    v = eval_outputs([t])

    assert v == 56, v
    assert isinstance(v, np.ndarray)
    assert v.shape == (), v.shape

    g = grad(t, s)
    assert eval_outputs([g]) == 0.0


def test_ScalarFromTensor():
    tt = constant(56)  # scal.constant(56)
    ss = scalar_from_tensor(tt)
    assert ss.owner.op is scalar_from_tensor
    assert ss.type.dtype == tt.type.dtype

    v = eval_outputs([ss])

    assert v == 56
    assert v.shape == ()

    if config.cast_policy == "custom":
        assert isinstance(v, np.int8)
    elif config.cast_policy in ("numpy", "numpy+floatX"):
        assert isinstance(v, str(np.asarray(56).dtype))
    else:
        raise NotImplementedError(config.cast_policy)

    tt = lscalar()
    ss = scalar_from_tensor(tt)
    ss.owner.op.grad([tt], [ss])
    fff = function([tt], ss)
    v = fff(np.asarray(5))
    assert v == 5
    assert isinstance(v, np.int64)
    assert v.shape == ()


class TestGrad:
    class Obj1(gof.op.Op):
        def __init__(self):
            self.gval0 = scalar("e")
            self.gval1 = scalar("f")

        def make_node(self):
            inputs = [scalar("a"), scalar("c")]
            outputs = [scalar("b"), scalar("d")]
            return gof.Apply(self, inputs, outputs)

        def grad(self, inp, grads):
            x0, x1 = inp
            gz0, gz1 = grads
            return self.gval0, self.gval1

    def test_1param(self):
        # grad: Test passing a single variable param
        o = TestGrad.Obj1()
        a1 = o.make_node()
        assert o.gval0 is tt.grad(a1.outputs[0], a1.inputs[0])

    def test_Nparam(self):
        # grad: Test passing multiple variable params
        o = TestGrad.Obj1()
        a1 = o.make_node()
        g0, g1 = grad(a1.outputs[0], a1.inputs)
        g0.name = None
        assert o.gval0 is g0
        assert o.gval1 is g1

    def test_grad_keep_type(self):
        # Tests that the theano grad method returns a list if it is passed a list
        # and a single variable if it is passed a single variable.
        # pylearn2 depends on theano behaving this way. This functionality has been
        # added three times and erroneously removed twice. If you do anything that
        # requires changing this test or making it fail you are almost certainly
        # making a common mistake, NOT fixing something.

        X = tt.matrix()
        y = X.sum()

        G = tt.grad(y, [X])

        assert isinstance(G, list)

        G = tt.grad(y, X)

        assert not isinstance(G, list)

    def test_1None_rval(self):
        # grad: Test returning a single zero value from grad
        o = TestGrad.Obj1()
        a1 = o.make_node()
        g = grad(a1.outputs[0], a1.outputs[1], disconnected_inputs="ignore")
        assert g.owner.op == fill
        assert g.owner.inputs[1].data == 0
        with pytest.raises(TypeError):
            grad(a1.outputs[0], "wtf")

    def test_NNone_rval(self):
        # grad: Test returning some zero value from grad
        o = TestGrad.Obj1()
        a1 = o.make_node()
        g0, g1, g2 = grad(
            a1.outputs[0], a1.inputs + [scalar("z")], disconnected_inputs="ignore"
        )
        assert o.gval0 is g0
        assert o.gval1 is g1
        assert g2.owner.op == fill
        assert g2.owner.inputs[1].data == 0

    def test_zero_gradient_shape(self):
        # Ensure that a zero gradient has the proper shape.
        x = dmatrix()
        f = theano.function([x], grad(dscalar(), x, disconnected_inputs="ignore"))
        a = np.ones((3, 7))
        assert (f(a) == 0).all()  # Zero gradient
        assert a.shape == f(a).shape  # With proper shape

    def test_cost_is_scalar(self):
        # grad: Test that a non-scalar cost raises a TypeError
        v = vector()
        m = matrix()
        # grad(v,...) and grad(m,...) should fail
        with pytest.raises(TypeError):
            grad(v, v)
        with pytest.raises(TypeError):
            grad(m, m)


class TestOpCache:
    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):
        # trigger bug in ticket #162
        v = matrix()
        v.name = "v"
        gv = fill(v / v, 1.0) / v - (fill(v / v, 1.0) * v) / (v * v)
        fn_py = inplace_func([v], gv)
        fn_c_or_py = inplace_func([v], gv)

        a = rand(5, 2).astype(config.floatX)
        assert np.all(fn_py(a) == fn_c_or_py(a))


class TestReshape(utt.InferShapeTester, utt.OptimizationTestMixin):
    def setup_method(self):
        self.shared = tt._shared
        self.op = Reshape
        # The tag canonicalize is needed for the shape test in FAST_COMPILE
        self.mode = None
        self.ignore_topo = (
            DeepCopyOp,
            opt.MakeVector,
            opt.Shape_i,
            DimShuffle,
            tt.Elemwise,
        )
        super().setup_method()

    def function(self, inputs, outputs, ignore_empty=False):
        f = function(inputs, outputs, mode=self.mode)
        if self.mode is not None or theano.config.mode != "FAST_COMPILE":
            topo = f.maker.fgraph.toposort()
            topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
            if ignore_empty:
                assert len(topo_) <= 1, topo_
            else:
                assert len(topo_) == 1, topo_
            if len(topo_) > 0:
                assert type(topo_[0].op) is self.op
        return f

    def test_reshape(self):
        a = dvector()
        b = dmatrix()
        d = dmatrix()

        # basic to 1 dim(without list)
        c = reshape(b, as_tensor_variable(6), ndim=1)
        f = self.function([b], c)

        b_val1 = np.asarray([[0, 1, 2], [3, 4, 5]])
        c_val1 = np.asarray([0, 1, 2, 3, 4, 5])
        b_val2 = b_val1.T
        c_val2 = np.asarray([0, 3, 1, 4, 2, 5])

        f_out1 = f(b_val1)
        f_out2 = f(b_val2)
        assert np.all(f_out1 == c_val1), (f_out1, c_val1)
        assert np.all(f_out2 == c_val2), (f_out2, c_val2)
        # print f.maker.fgraph.toposort()
        # check that we remove the useless reshape

        # basic to 1 dim(with list)
        c = reshape(b, (as_tensor_variable(6),), ndim=1)
        f = self.function([b], c)
        assert np.all(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])) == np.asarray([0, 1, 2, 3, 4, 5])
        )
        # print f.maker.fgraph.toposort()
        # check that we remove the useless reshape

        # basic to shape object of same ndim
        c = reshape(b, d.shape)
        f = self.function([b, d], c)
        assert np.all(
            f(np.asarray([[0, 1, 2], [3, 4, 5]]), [[0, 1], [2, 3], [4, 5]])
            == np.asarray([[0, 1], [2, 3], [4, 5]])
        )

        # basic to 2 dims
        c = reshape(a, [2, 3])
        f = self.function([a], c)
        assert np.all(
            f(np.asarray([0, 1, 2, 3, 4, 5])) == np.asarray([[0, 1, 2], [3, 4, 5]])
        )

        # test that it works without inplace operations
        a_val = np.asarray([0, 1, 2, 3, 4, 5])
        a_val_copy = np.asarray([0, 1, 2, 3, 4, 5])
        b_val = np.asarray([[0, 1, 2], [3, 4, 5]])

        f_sub = self.function([a, b], c - b)
        assert np.all(f_sub(a_val, b_val) == 0.0)
        assert np.all(a_val == a_val_copy)

        # test that it works with inplace operations
        a_val = theano._asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        a_val_copy = theano._asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        b_val = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")

        f_sub = self.function([a, b], c - b)
        assert np.all(f_sub(a_val, b_val) == 0.0)
        assert np.all(a_val == a_val_copy)

        # verify gradient
        def just_vals(v):
            return Reshape(2)(v, theano._asarray([2, 3], dtype="int32"))

        utt.verify_grad(just_vals, [a_val], mode=self.mode)

        # test infer_shape
        self._compile_and_check([a], [c], (a_val,), self.op)

        # test broadcast flag for constant value of 1
        c = reshape(b, (b.shape[0], b.shape[1], 1))
        # That reshape may get replaced with a dimshuffle, with is ignored,
        # so we pass "ignore_empty=True"
        f = self.function([b], c, ignore_empty=True)
        assert np.all(
            f(np.asarray([[0, 1, 2], [3, 4, 5]]))
            == np.asarray([[[0], [1], [2]], [[3], [4], [5]]])
        )
        assert f.maker.fgraph.toposort()[-1].outputs[0].type.broadcastable == (
            False,
            False,
            True,
        )

        # test broadcast flag for constant value of 1 if it cannot be
        # replaced with dimshuffle
        c = reshape(b, (b.shape[1], b.shape[0], 1))
        f = self.function([b], c, ignore_empty=True)
        assert np.all(
            f(np.asarray([[0, 1, 2], [3, 4, 5]]))
            == np.asarray([[[0], [1]], [[2], [3]], [[4], [5]]])
        )
        assert f.maker.fgraph.toposort()[-1].outputs[0].type.broadcastable == (
            False,
            False,
            True,
        )

    def test_m1(self):
        t = tensor3()
        rng = np.random.RandomState(seed=utt.fetch_seed())
        val = rng.uniform(size=(3, 4, 5)).astype(config.floatX)
        for out in [
            t.reshape([-1]),
            t.reshape([-1, 5]),
            t.reshape([5, -1]),
            t.reshape([5, -1, 3]),
        ]:
            self._compile_and_check([t], [out], [val], self.op)

    def test_reshape_long_in_shape(self):
        v = dvector("v")
        r = v.reshape((v.shape[0], 1))
        print(r.eval({v: np.arange(5.0)}))
        assert np.allclose(r.eval({v: np.arange(5.0)}).T, np.arange(5.0))

    def test_bad_shape(self):
        a = matrix("a")
        shapes = ivector("shapes")
        rng = np.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.uniform(size=(3, 4)).astype(config.floatX)

        # Test reshape to 1 dim
        r = a.reshape(shapes, ndim=1)

        f = self.function([a, shapes], r)
        with pytest.raises(ValueError):
            f(a_val, [13])

        # Test reshape to 2 dim
        r = a.reshape(shapes, ndim=2)

        f = self.function([a, shapes], r)

        with pytest.raises(ValueError):
            f(a_val, [-1, 5])
        with pytest.raises(ValueError):
            f(a_val, [7, -1])
        with pytest.raises(ValueError):
            f(a_val, [7, 5])
        with pytest.raises(ValueError):
            f(a_val, [-1, -1])

    def test_0(self):
        x = fvector("x")
        f = self.function([x], x.reshape((0, 100)))
        assert f(np.ndarray((0,), dtype="float32")).shape == (0, 100)

    def test_empty_shp(self):
        const = tt.constant([1]).reshape(())
        f = function([], const)
        assert f().shape == ()


def test_make_column_matrix_broadcastable():
    # The goal of the operation made by `b` is to ensure the second dimension
    # of the column matrix is broadcastable.
    a = tt.dmatrix()
    b = a.reshape((a.shape[0],)).dimshuffle(0, "x")
    f = function([a], b)
    assert (f(np.zeros((3, 1))) + np.ones(2) == np.ones((3, 2))).all()


def test_flatten_outdimNone():
    a = dmatrix()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = theano._asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_scalar():
    a = dscalar()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = theano._asarray(3.0, dtype="float64")
    c_val = theano._asarray([3.0], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    # utt.verify_grad(flatten, [a_val]) #TODO: fix verify_grd to work on scalars


def test_flatten_ndim1():
    a = dmatrix()
    c = flatten(a, 1)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = theano._asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_ndim2():
    a = dmatrix()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    assert np.all(f(a_val) == a_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == a_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_ndim2_of_3():
    a = TensorType("float64", (False, False, False))()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = theano._asarray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="float64")
    c_val = theano._asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])
    # test outdim parameter name
    flatten_2 = partial(flatten, outdim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_broadcastable():
    # Ensure that the broadcastable pattern of the output is coherent with
    # that of the input

    inp = TensorType("float64", (False, False, False, False))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, False, False, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, True, False, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, False)

    inp = TensorType("float64", (False, True, True, True))()
    out = flatten(inp, ndim=2)
    assert out.broadcastable == (False, True)

    inp = TensorType("float64", (True, False, True, True))()
    out = flatten(inp, ndim=3)
    assert out.broadcastable == (True, False, True)


def test_flatten_ndim_invalid():
    a = dmatrix()
    with pytest.raises(ValueError):
        flatten(a, 3)
    with pytest.raises(ValueError):
        flatten(a, 0)


def test_is_flat():
    # tests is_flat method for constant and symbolic variables,
    # as well as reshaped constant and symbolic variables on the
    # given outdim

    # Constant variable
    assert tt.is_flat(tt.as_tensor_variable(np.zeros(10)))
    assert tt.is_flat(tt.as_tensor_variable(np.zeros((10, 10, 10))), ndim=3)
    assert not tt.is_flat(tt.as_tensor_variable(np.zeros((10, 10, 10))))

    # Symbolic variable
    assert tt.is_flat(tt.vector())
    assert tt.is_flat(tt.tensor3(), ndim=3)
    assert not tt.is_flat(tt.tensor3())

    # Reshape with constant shape
    X = tt.tensor4()
    assert tt.is_flat(X.reshape((-1,)))
    assert tt.is_flat(X.reshape((10, 10, -1)), ndim=3)
    assert not tt.is_flat(X.reshape((10, 10, -1)))

    # Reshape with symbolic shape
    X = tt.tensor4()
    assert tt.is_flat(X.reshape((tt.iscalar(),)))
    assert tt.is_flat(X.reshape((tt.iscalar(),) * 3), ndim=3)
    assert not tt.is_flat(X.reshape((tt.iscalar(),) * 3))


def test_tile():
    def run_tile(x, x_, reps, use_symbolic_reps):
        if use_symbolic_reps:
            rep_symbols = [iscalar() for _ in range(len(reps))]
            f = function([x] + rep_symbols, tile(x, rep_symbols))
            return f(*([x_] + list(reps)))
        else:
            f = function([x], tile(x, reps))
            return f(x_)

    rng = np.random.RandomState(utt.fetch_seed())

    for use_symbolic_reps in [False, True]:
        # Test the one-dimensional case.
        x = vector()
        x_ = rng.randn(5).astype(config.floatX)
        assert np.all(run_tile(x, x_, (2,), use_symbolic_reps) == np.tile(x_, (2,)))

        # Test the two-dimensional case.
        x = matrix()
        x_ = rng.randn(2, 4).astype(config.floatX)
        assert np.all(run_tile(x, x_, (2, 3), use_symbolic_reps) == np.tile(x_, (2, 3)))

        # Test the three-dimensional case.
        x = tensor3()
        x_ = rng.randn(2, 4, 3).astype(config.floatX)
        assert np.all(
            run_tile(x, x_, (2, 3, 4), use_symbolic_reps) == np.tile(x_, (2, 3, 4))
        )

        # Test the four-dimensional case.
        x = tensor4()
        x_ = rng.randn(2, 4, 3, 5).astype(config.floatX)
        assert np.all(
            run_tile(x, x_, (2, 3, 4, 6), use_symbolic_reps)
            == np.tile(x_, (2, 3, 4, 6))
        )

    # Test when reps is integer, tt.scalar or tt.vector.
    # Test 1,2,3,4-dimensional cases.
    # Test input x has the shape [2], [2, 4], [2, 4, 3], [2, 4, 3, 5].
    test_shape = [2, 4, 3, 5]
    k = 0
    for xtype in [vector(), matrix(), tensor3(), tensor4()]:
        x = xtype
        k = k + 1
        x_ = rng.randn(*test_shape[0:k]).astype(config.floatX)

        # integer:
        reps_ = 2
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))

        # tt.scalar:
        reps = iscalar()
        reps_ = 2
        f = function([x, reps], tile(x, reps))
        assert np.all(f(x_, reps_) == np.tile(x_, reps_))

        # tt.vector:
        reps = ivector()
        reps_ = [2] if k == 1 or k == 2 else [2, 3]
        ndim_ = k
        f = function([x, reps], tile(x, reps, ndim_))
        assert np.all(f(x_, reps_) == np.tile(x_, reps_))

        # list of integers:
        reps_ = [2, 3, 4]
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))

        # list of integers and tt.scalars:
        d = iscalar()
        reps = [2, d, 4]
        f = function([x, d], tile(x, reps))
        reps_ = [2, 3, 4]
        assert np.all(f(x_, 3) == np.tile(x_, reps_))

        # reps is list, len(reps) > x.ndim, 3 cases below:
        r = [2, 3, 4, 5, 6]
        reps_ = r[: k + 1]  # len(reps_) = x.ndim+1
        # (1) ndim = None.
        f = function([x], tile(x, reps_))
        assert np.all(f(x_) == np.tile(x_, reps_))
        # (2) ndim = len(reps).
        ndim_ = len(reps_)
        f = function([x], tile(x, reps_, ndim_))
        assert np.all(f(x_) == np.tile(x_, reps_))
        # (3) ndim > len(reps)
        ndim_ = len(reps_) + 1
        f = function([x], tile(x, reps_, ndim_))
        assert np.all(f(x_) == np.tile(x_, [1] + reps_))

        # reps is list, ndim > x.ndim > len(reps):
        r = [2, 3, 4, 5]
        if k > 1:
            ndim_ = k + 1
            reps_ = r[: k - 1]
            f = function([x], tile(x, reps_, ndim_))
            assert np.all(f(x_) == np.tile(x_, [1, 1] + reps_))

        # error raising test: ndim not specified when reps is vector
        reps = ivector()
        with pytest.raises(ValueError):
            tile(x, reps)

        # error raising test: not a integer
        for reps in [2.5, fscalar(), fvector()]:
            with pytest.raises(ValueError):
                tile(x, reps)

        # error raising test: the dimension of reps exceeds 1
        reps = imatrix()
        with pytest.raises(ValueError):
            tile(x, reps)

        # error raising test: ndim is not None, ndim < x.ndim
        # 3 cases below (reps is list/tt.scalar/tt.vector):
        for reps in [[2, 3, 4], iscalar(), ivector()]:
            if k > 1:
                ndim = k - 1
                with pytest.raises(ValueError):
                    tile(x, reps, ndim)

        # error raising test: reps is list, len(reps) > ndim
        r = [2, 3, 4, 5, 6]
        reps = r[: k + 1]
        ndim = k
        with pytest.raises(ValueError):
            tile(x, reps, ndim)

        # error raising test:
        # reps is tt.vector and len(reps_value) > ndim,
        # reps_value is the real value when excuting the function.
        reps = ivector()
        r = [2, 3, 4, 5, 6, 7]
        reps_ = r[: k + 2]
        ndim_ = k + 1
        f = function([x, reps], tile(x, reps, ndim_))
        with pytest.raises(AssertionError):
            f(x_, reps_)


def test_tile_grad():
    def grad_tile(x, reps, np_x):
        y = tile(x, reps)
        z = y.sum()
        g = theano.function([x], grad(z, x))
        grad_res = g(np_x)
        # The gradient should be the product of the tiling dimensions
        # (since the gradients are additive through the tiling operation)
        assert np.all(grad_res == np.prod(reps))

    rng = np.random.RandomState(utt.fetch_seed())

    # test vector
    grad_tile(vector("x"), [3], rng.randn(5).astype(config.floatX))
    # test matrix
    grad_tile(matrix("x"), [3, 4], rng.randn(2, 3).astype(config.floatX))
    # test tensor3
    grad_tile(tensor3("x"), [3, 4, 5], rng.randn(2, 4, 3).astype(config.floatX))
    # test tensor4
    grad_tile(tensor4("x"), [3, 4, 5, 6], rng.randn(2, 4, 3, 5).astype(config.floatX))


class TestARange:
    def setup_method(self):
        utt.seed_rng()

    def test_Op_integers(self):
        # Test behaviour of ARange Op on integer inputs
        start, stop, step = iscalars("start", "stop", "step")
        out = ARange(start.type.dtype)(start, stop, step)
        f = function([start, stop, step], out)

        assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
        assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
        assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
        assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
        assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
        assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    def test_grads(self):
        def f(start, stop, step):
            return ARange(start.type.dtype)(start, stop, step)

        rng = np.random.RandomState(utt.fetch_seed())
        # Due to the random projection, we should not use the exact
        # point that change the shape of the output.
        for start, stop, step in [(0, 4.9, 1), (5.1, 0, -0.5), (1, 5.1, 0.5)]:
            utt.verify_grad(
                f,
                [
                    np.asarray(start).astype(config.floatX),
                    np.asarray(stop).astype(config.floatX),
                    np.asarray(step).astype(config.floatX),
                ],
                rng=rng,
            )

    def test_integers(self):
        # Test arange constructor, on integer outputs
        start, stop, step = iscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(np.array(1, dtype="int32")).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
        assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
        assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
        assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
        assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
        assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    def test_float32(self):
        # Test arange constructor, on float32 outputs
        start, stop, step = fscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        if config.cast_policy == "custom":
            assert out.dtype == start.type.dtype
        elif config.cast_policy == "numpy":
            numpy_dtype = np.arange(
                np.array(0, dtype=start.dtype),
                np.array(1, dtype=stop.dtype),
                np.array(1, dtype=step.dtype),
            ).dtype
            assert out.dtype == numpy_dtype
        elif config.cast_policy == "numpy+floatX":
            assert out.dtype == config.floatX
        else:
            raise NotImplementedError(config.cast_policy)
        arg_vals = [(0, 5, 1), (2, 11, 4), (-5, 1.1, 1.2), (1.3, 2, -2.1), (10, 2, 2)]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == "custom":
                expected_val = np.arange(
                    start_v, stop_v, step_v, dtype=start.type.dtype
                )
            elif config.cast_policy in ("numpy", "numpy+floatX"):
                expected_val = np.arange(start_v_, stop_v_, step_v_, dtype=out.dtype)
            else:
                raise NotImplementedError(config.cast_policy)
            assert np.all(f_val == expected_val)

    def test_float64(self):
        # Test arange constructor, on float64 outputs
        start, stop, step = dscalars("start", "stop", "step")
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        arg_vals = [(0, 5, 1), (2, 11, 4), (-5, 1.1, 1.2), (1.3, 2, -2.1), (10, 2, 2)]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == "custom":
                expected_val = np.arange(
                    start_v, stop_v, step_v, dtype=start.type.dtype
                )
            elif config.cast_policy in ("numpy", "numpy+floatX"):
                expected_val = np.arange(start_v_, stop_v_, step_v_)
            else:
                raise NotImplementedError(config.cast_policy)
            assert np.all(f_val == expected_val)

    def test_default_step(self):
        # Test that arange constructor uses the correct default step
        start, stop = iscalars("start", "stop")
        out = arange(start, stop)
        f = function([start, stop], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(0), np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5) == np.arange(0, 5))
        assert np.all(f(-5, 1) == np.arange(-5, 1))
        assert np.all(f(0, 0) == np.arange(0, 0))

        dstart, dstop = dscalars("start", "stop")
        dout = arange(dstart, dstop)
        df = function([dstart, dstop], dout)

        assert dout.dtype == dstart.type.dtype
        # print df(0.2, 5.3)
        # print np.arange(0.2, 5.3)
        assert np.all(df(0.2, 5.3) == np.arange(0.2, 5.3))
        assert np.all(df(0.8, 5.3) == np.arange(0.8, 5.3))
        assert np.all(df(-0.7, 5.3) == np.arange(-0.7, 5.3))

    def test_default_start(self):
        # Test that arange constructor uses the correct default start
        stop = iscalar("stop")
        out = arange(stop)
        f = function([stop], out)

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(8) == np.arange(8))
        assert np.all(f(-2) == np.arange(-2))

        fstop = fscalar("stop")
        fout = arange(fstop)
        ff = function([fstop], fout)

        if config.cast_policy == "custom":
            assert fout.dtype == fstop.type.dtype
        elif config.cast_policy == "numpy":
            assert fout.dtype == np.arange(np.float32(1)).dtype
        elif config.cast_policy == "numpy+floatX":
            if config.floatX == "float32":
                assert fout.dtype == "float32"
            else:
                assert fout.dtype == np.arange(np.float32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)

        fstop_values = [0.2, -0.7, 8.5]
        for fstop_v in fstop_values:
            fstop_v32 = np.float32(fstop_v)
            assert np.all(ff(fstop_v32) == np.arange(fstop_v))

    def test_upcast(self):
        # Test that arange computes output type adequately
        if config.cast_policy == "custom":
            assert arange(iscalar()).dtype == "int64"
            assert arange(fscalar()).dtype == fscalar().dtype
            assert arange(dscalar()).dtype == dscalar().dtype

            # int32 + float32 -> float64
            assert arange(iscalar(), fscalar()).dtype == dscalar().dtype
            assert arange(iscalar(), dscalar()).dtype == dscalar().dtype
            assert arange(fscalar(), dscalar()).dtype == dscalar().dtype

            assert arange(iscalar(), fscalar(), dscalar()).dtype == dscalar().dtype
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            for dtype in get_numeric_types():
                # Test with a single argument.
                arange_dtype = arange(scalar(dtype=str(dtype))).dtype
                numpy_dtype = np.arange(np.array(1, dtype=dtype)).dtype
                if (
                    dtype != "float64"
                    and numpy_dtype == "float64"
                    and config.cast_policy == "numpy+floatX"
                    and config.floatX == "float32"
                ):
                    # We want a float32 arange.
                    assert arange_dtype == "float32"
                else:
                    # Follow numpy.
                    assert arange_dtype == numpy_dtype

                # Test with two arguments.
                for stop_dtype in get_numeric_types():
                    arange_dtype = arange(
                        start=scalar(dtype=str(dtype)),
                        stop=scalar(dtype=str(stop_dtype)),
                    ).dtype
                    numpy_dtype = np.arange(
                        start=np.array(0, dtype=dtype),
                        stop=np.array(1, dtype=stop_dtype),
                    ).dtype
                    if (
                        dtype != "float64"
                        and stop_dtype != "float64"
                        and numpy_dtype == "float64"
                        and config.cast_policy == "numpy+floatX"
                        and config.floatX == "float32"
                    ):
                        # We want a float32 arange.
                        assert arange_dtype == "float32"
                    else:
                        # Follow numpy.
                        assert arange_dtype == numpy_dtype

                    # Test with three arguments.
                    for step_dtype in get_numeric_types():
                        arange_dtype = arange(
                            start=scalar(dtype=str(dtype)),
                            stop=scalar(dtype=str(stop_dtype)),
                            step=scalar(dtype=str(step_dtype)),
                        ).dtype
                        numpy_dtype = np.arange(
                            start=np.array(0, dtype=dtype),
                            stop=np.array(1, dtype=stop_dtype),
                            step=np.array(1, dtype=step_dtype),
                        ).dtype
                        if (
                            dtype != "float64"
                            and stop_dtype != "float64"
                            and step_dtype != "float64"
                            and numpy_dtype == "float64"
                            and config.cast_policy == "numpy+floatX"
                            and config.floatX == "float32"
                        ):
                            # We want a float32 arange.
                            assert arange_dtype == "float32"
                        else:
                            # Follow numpy.
                            assert arange_dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

    def test_dtype_cache(self):
        # Checks that the same Op is returned on repeated calls to arange
        # using the same dtype, but not for different dtypes.

        start, stop, step = iscalars("start", "stop", "step")
        out1 = arange(start, stop, step)
        out2 = arange(start, stop, step, dtype=out1.dtype)
        out3 = arange(start, stop, 2.0, dtype=out1.dtype)
        out4 = arange(start, stop, 2.0)

        assert out1.owner.op is out2.owner.op
        assert out2.owner.op is out3.owner.op
        assert out3.owner.op is not out4.owner.op

    def test_infer_shape(self):
        start, stop, step = iscalars("start", "stop", "step")
        out = arange(start, stop, step)
        mode = theano.config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = compile.mode.get_mode(mode).excluding("fusion")
        f = function([start, stop, step], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 9

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(
                np.array(0, dtype=start.dtype),
                np.array(1, dtype=stop.dtype),
                np.array(1, dtype=step.dtype),
            ).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

        assert np.all(f(0, 5, 1) == len(np.arange(0, 5, 1)))
        assert np.all(f(2, 11, 4) == len(np.arange(2, 11, 4)))
        assert np.all(f(-5, 1, 1) == len(np.arange(-5, 1, 1)))
        assert np.all(f(10, 2, -2) == len(np.arange(10, 2, -2)))
        assert np.all(f(10, 2, 2) == len(np.arange(10, 2, 2)))
        assert np.all(f(0, 0, 1) == len(np.arange(0, 0, 1)))

        out = arange(start, stop, 1)
        f = function([start, stop], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 5
        # 4 [Elemwise{sub,no_inplace}(stop, start), Elemwise{Cast{int64}}(Elemwise{sub,no_inplace}.0), Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{int64}}.0, 0), MakeVector(Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)].0)]
        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            assert out.dtype == np.arange(np.int32(0), np.int32(1), np.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert np.all(f(0, 5) == len(np.arange(0, 5)))
        assert np.all(f(2, 11) == len(np.arange(2, 11)))
        assert np.all(f(-5, 1) == len(np.arange(-5, 1)))
        assert np.all(f(10, 2) == len(np.arange(10, 2)))
        assert np.all(f(10, 2) == len(np.arange(10, 2)))
        assert np.all(f(0, 0) == len(np.arange(0, 0)))
        assert np.all(f(-64, 64) == len(np.arange(-64, 64)))
        assert arange(-64, 64).shape.eval() == [128]
        assert arange(-64, 64, 2).shape.eval() == [64]

        out = arange(0, stop, 1)
        f = function([stop], out.shape, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 2
        # [Elemwise{Cast{int64}}(stop), MakeVector(Elemwise{Cast{int64}}.0)]

        if config.cast_policy == "custom":
            assert out.dtype == "int64"
        elif config.cast_policy in ("numpy", "numpy+floatX"):
            numpy_dtype = np.arange(0, np.array(1, dtype=stop.dtype), 1).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

        assert np.all(f(5) == len(np.arange(0, 5)))
        assert np.all(f(11) == len(np.arange(0, 11)))
        assert np.all(f(1) == len(np.arange(0, 1)))
        assert np.all(f(2) == len(np.arange(0, 2)))
        assert np.all(f(2) == len(np.arange(0, 2)))
        assert np.all(f(0) == len(np.arange(0, 0)))


class TestNdGrid:
    def setup_method(self):
        pass

    def test_mgrid_numpy_equiv(self):
        nmgrid = (
            [np.mgrid[0:1:0.1]],
            np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        tmgrid = (
            [mgrid[0:1:0.1]],
            mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nmgrid, tmgrid):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg.eval())

    def test_ogrid_numpy_equiv(self):
        nogrid = (
            [np.ogrid[0:1:0.1]],
            np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        togrid = (
            [ogrid[0:1:0.1]],
            ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nogrid, togrid):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg.eval())

    def test_mgrid_theano_variable_numpy_equiv(self):
        nfmgrid = np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        nimgrid = np.mgrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfmgrid = mgrid[i:1:0.1, 1:j:1.0, 10:100:k]
        timgrid = mgrid[l:2:1, 1:m:1, 10:100:n]
        ff = theano.function([i, j, k], tfmgrid)
        fi = theano.function([l, m, n], timgrid)
        for n, t in zip((nfmgrid, nimgrid), (ff(0, 10, 10.0), fi(0, 10, 10))):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg)

    def test_ogrid_theano_variable_numpy_equiv(self):
        nfogrid = np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        niogrid = np.ogrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfogrid = ogrid[i:1:0.1, 1:j:1.0, 10:100:k]
        tiogrid = ogrid[l:2:1, 1:m:1, 10:100:n]
        ff = theano.function([i, j, k], tfogrid)
        fi = theano.function([l, m, n], tiogrid)
        for n, t in zip((nfogrid, niogrid), (ff(0, 10, 10.0), fi(0, 10, 10))):
            for ng, tg in zip(n, t):
                utt.assert_allclose(ng, tg)


class TestInversePermutation:
    def setup_method(self):
        utt.seed_rng()

    def test_dim1(self):
        # Test the inversion of one permutation (int vector)
        p = ivector()
        inv = inverse_permutation(p)
        assert inv.dtype == p.dtype
        f_inverse = function([p], inv)

        # Generate a random permutation
        rng = np.random.RandomState(utt.fetch_seed())
        p_val = rng.permutation(10).astype("int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that permutation(inverse) == inverse(permutation) = identity
        assert np.all(p_val[inv_val] == np.arange(10))
        assert np.all(inv_val[p_val] == np.arange(10))

    def test_dim2(self):
        # Test the inversion of several permutations at a time
        # Each row of p is a different permutation to inverse
        p = imatrix()
        inv = inverse_permutation(p)
        f_inverse = function([p], inv)

        rng = np.random.RandomState(utt.fetch_seed())
        # Generate 10 random permutations
        p_val = np.asarray([rng.permutation(10) for i in range(7)], dtype="int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation list
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that, for each permutation,
        # permutation(inverse) == inverse(permutation) = identity
        for p_row, i_row in zip(p_val, inv_val):
            assert np.all(p_row[i_row] == np.arange(10))
            assert np.all(i_row[p_row] == np.arange(10))


class TestPermuteRowElements:
    def setup_method(self):
        utt.seed_rng()

    def test_1_1(self):
        # Test PermuteRowElements(vector, vector)
        input = dvector()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(5,))
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # Should be equivalent to advanced indexing
        out_bis = input_val[p_val]
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_1(self):
        # Test broadcasting in PermuteRowElements(matrix, vector)
        input = matrix()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # The same permutation should be applied to every row of the input matrix.
        out_bis = np.asarray([r[p_val] for r in input_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_2(self):
        # Test PermuteRowElements(matrix, matrix)
        input = matrix()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the corresponding
        # row of input
        out_bis = np.asarray([i_row[p_row] for i_row, p_row in zip(input_val, p_val)])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_1_2(self):
        # Test PermuteRowElements(vector, matrix)
        # Different permutations will be applied to the same input vector
        input = vector()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(5,)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the input vector
        out_bis = np.asarray([input_val[p_row] for p_row in p_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_3b_2(self):
        # Test permute_row_elements on a more complex broadcasting pattern:
        # input.type.broadcastable = (False, True, False),
        # p.type.broadcastable = (False, False).

        input = TensorType("floatX", (False, True, False))()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(4, 1, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to each row
        # of the input tensor
        out_bis = np.asarray(
            [[in_mat[0, p_row] for p_row in p_val] for in_mat in input_val]
        )
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])


class TestTensordot:
    def TensorDot(self, axes):
        # Since tensordot is no longer an op, mimic the old op signature
        # to allow easy use of verify_grad.
        return lambda a, b: tensordot(a, b, axes)

    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):

        # Test vector-vector
        avec = vector()
        bvec = vector()
        axes = ((0,), (0,))
        c = tensordot(avec, bvec, axes)
        f1 = inplace_func([avec, bvec], c)
        aval = rand(5)
        bval = rand(5)
        out0 = np.tensordot(aval, bval, axes)
        out1 = f1(aval, bval)
        utt.assert_allclose(out0, out1)
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test matrix-vector
        bmat = matrix()
        axes = ((0,), (1,))
        c = tensordot(avec, bmat, axes)
        f2 = inplace_func([avec, bmat], c)
        aval = rand(5)
        bval = rand(8, 5)
        utt.assert_allclose(np.tensordot(aval, bval, axes), f2(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test matrix-matrix
        amat = matrix()
        for axes, shps in [
            [((0,), (0,)), [(4, 7), (4, 9)]],
            [((0,), (1,)), [(4, 7), (9, 4)]],
            [((1,), (0,)), [(4, 7), (7, 9)]],
            [((1,), (1,)), [(4, 7), (9, 7)]],
            [((0, 1), (0, 1)), [(4, 7), (4, 7)]],
            # [((0, 1), (1, 0)), [(4, 7), (7, 4)]],
            # [((1, 0), (1, 0)), [(4, 7), (4, 7)]],
            # [((1, 0), (0, 1)), [(4, 7), (7, 4)]],
        ]:
            c = tensordot(amat, bmat, axes)
            f3 = inplace_func([amat, bmat], c)
            aval = rand(*shps[0])
            bval = rand(*shps[1])
            utt.assert_allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test ndarray-matrix, sum over one dim of matrix
        for axes, shps in [
            [((2,), (1,)), [(1, 2, 3, 4), (2, 3)]],
            [((0,), (1,)), [(1, 2, 3, 4), (3, 1)]],
            [((0,), (0,)), [(1, 2, 3, 4), (1, 3)]],
            [((3,), (0,)), [(1, 2, 3, 4), (4, 1)]],
            # [((3, 1), (0, 1)), [(1, 2, 3, 4), (4, 2)]],
            # [((0, 1), (1, 0)), [(1, 2, 3, 4), (2, 1)]],
            # [((3, 1), (1, 0)), [(1, 2, 3, 4), (2, 4)]],
        ]:
            atens = tensor4()
            c = tensordot(atens, bmat, axes)
            f4 = inplace_func([atens, bmat], c)
            aval = rand(*shps[0])
            bval = rand(*shps[1])
            utt.assert_allclose(np.tensordot(aval, bval, axes), f4(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test ndarray-ndarray
        atens = tensor4()
        btens = tensor3()
        axes = ((1, 3), (0, 2))
        c = tensordot(atens, btens, axes)
        f5 = inplace_func([atens, btens], c)
        aval = rand(4, 3, 5, 2)
        bval = rand(3, 4, 2)
        utt.assert_allclose(np.tensordot(aval, bval, axes), f5(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        axes = (axes[1], axes[0])
        c = tensordot(btens, atens, axes)
        f6 = inplace_func([btens, atens], c)
        utt.assert_allclose(np.tensordot(bval, aval, axes), f6(bval, aval))
        utt.verify_grad(self.TensorDot(axes), [bval, aval])

    def test_raise_error(self):
        amat = matrix()
        bmat = matrix()
        bvec = vector()

        # Test invalid length for axes
        with pytest.raises(ValueError):
            tensordot(amat, bmat, (0, 1, 2))

        # Test axes of uneven length
        with pytest.raises(ValueError):
            tensordot(amat, bmat, ((0, 1), (0)))

        # Test invalid len(axes) given inputs are matrices
        with pytest.raises(ValueError):
            tensordot(amat, bmat, ((0, 1, 2), (0, 1, 2)))

        # Test invalid axes[1] given that y is a vector
        with pytest.raises(ValueError):
            tensordot(amat, bvec, (0, 1))

        # Test invalid scalar axes given inputs are matrices
        with pytest.raises(ValueError):
            tensordot(amat, bvec, 2)

    def test_weird_valid_axes(self):
        # Test matrix-matrix
        amat = matrix()
        bmat = matrix()
        for axes in [0, (1, 0), [1, 0], (1, (0,)), ((1,), 0), ([1], [0]), ([], [])]:
            c = tensordot(amat, bmat, axes)
            f3 = inplace_func([amat, bmat], c)
            aval = rand(4, 7)
            bval = rand(7, 9)
            utt.assert_allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_scalar_axes(self):
        # Test matrix-matrix
        amat = fmatrix()
        bmat = dmatrix()
        # We let at float64 to test mix of float32 and float64.
        axes = 1
        aval = rand(4, 5).astype("float32")
        bval = rand(5, 3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test tensor-tensor
        amat = tensor3()
        bmat = tensor3()
        axes = 2
        aval = rand(3, 4, 5)
        bval = rand(4, 5, 3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_scalar0(self):
        # Test tensor-tensor
        amat = matrix()
        bmat = matrix()
        axes = 0
        aval = rand(4, 5)
        bval = rand(5, 4)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_broadcastable1(self):
        x = TensorType(dtype=config.floatX, broadcastable=(True, False, False))("x")
        y = tensor3("y")
        z = tensordot(x, y)
        assert z.broadcastable == (True, False)
        f = inplace_func([x, y], z)
        xv = rand(1, 3, 4)
        yv = rand(3, 4, 5)
        zv = f(xv, yv)
        assert np.allclose(np.tensordot(xv, yv), zv)

    def test_broadcastable2(self):
        x = TensorType(dtype=config.floatX, broadcastable=(True, False, False))("x")
        y = tensor3("y")
        axes = [[2, 1], [0, 1]]
        z = tensordot(x, y, axes=axes)
        assert z.broadcastable == (True, False)
        f = inplace_func([x, y], z)
        xv = rand(1, 3, 4)
        yv = rand(4, 3, 5)
        zv = f(xv, yv)
        assert np.allclose(np.tensordot(xv, yv, axes=axes), zv)


def test_smallest_stack():
    sx, sy = dscalar(), dscalar()

    rval = inplace_func([sx, sy], stack([sx, sy]))(-4.0, -2.0)
    assert type(rval) == np.ndarray
    assert [-4, -2] == list(rval)


def test_smallest():
    x = dvector()
    y = dvector()
    z = dvector()
    f1 = inplace_func([x], smallest(x))
    assert np.all([1, 2, 3] == f1([1, 2, 3]))
    f3 = inplace_func([x, y, z], smallest(x, y, z))
    assert np.all([1, 2, 3] == f3([1, 3, 9], [7, 7, 7], [8, 2, 3]))

    sx, sy = dscalar(), dscalar()

    assert -4 == inplace_func([sx, sy], smallest(sx, sy))(-4.0, -2.0)


def test_reshape_member_fn():
    x = dmatrix()
    y = x.reshape((4, 5, 6))
    assert y.owner.op == Reshape(3)


def test_var():
    a = Tensor(dtype="float64", broadcastable=[False, False, False])()
    f = function([a], var(a))

    a_val = np.arange(60).reshape(3, 4, 5)
    assert np.allclose(np.var(a_val), f(a_val))

    f = function([a], var(a, axis=0))
    assert np.allclose(np.var(a_val, axis=0), f(a_val))

    f = function([a], var(a, axis=1))
    assert np.allclose(np.var(a_val, axis=1), f(a_val))

    f = function([a], var(a, axis=2))
    assert np.allclose(np.var(a_val, axis=2), f(a_val))

    f = function([a], var(a, axis=0, ddof=0))
    assert np.allclose(np.var(a_val, axis=0, ddof=0), f(a_val))

    f = function([a], var(a, axis=1, ddof=1))
    assert np.allclose(np.var(a_val, axis=1, ddof=1), f(a_val))

    f = function([a], var(a, axis=2, ddof=1))
    assert np.allclose(np.var(a_val, axis=2, ddof=1), f(a_val))

    f = function([a], var(a, ddof=0, corrected=True))
    mean_a = np.mean(a_val)
    centered_a = a_val - mean_a
    v = np.mean(centered_a ** 2)
    error = (np.mean(centered_a)) ** 2
    v = v - error
    assert np.allclose(v, f(a_val))

    f = function([a], var(a, axis=2, ddof=1, corrected=True))
    mean_a = np.mean(a_val, axis=2, keepdims=True)
    centered_a = a_val - mean_a
    v = np.var(a_val, axis=2, ddof=1)
    shp_inp = np.shape(a_val)
    shp = shp_inp - np.array(1)
    error = (np.sum(centered_a, axis=2)) ** 2
    error = np.true_divide(error, shp[1] * shp_inp[1])
    v = v - error
    assert np.allclose(v, f(a_val))

    # Test that we don't upcast float16 computation
    assert tt.vector(dtype="float16").var().dtype == "float16"


class TestSum:
    def test_sum_overflow(self):
        # Ensure that overflow errors are a little bit harder to get
        a = Tensor(dtype="int8", broadcastable=[False])()
        f = function([a], sum(a))
        assert f([1] * 300) == 300

    def test_list(self):
        ll = [theano.shared(0.0), theano.shared(2.0)]
        tt.sum(ll).eval() == 2


@pytest.mark.skipif(
    isinstance(get_default_mode(), theano.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default():
    x, y = scalars("xy")
    z = default(x, y)
    f = function([x, y], z)
    assert f(1, 2) == 1
    assert f(None, 2) == 2
    assert f(1, None) == 1


@pytest.mark.skipif(
    isinstance(get_default_mode(), theano.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default_state():
    x, y = scalars("xy")
    # print config.floatX
    # print x.type
    # print y.type
    z = default(x, 3.8)
    new_x = y + z
    f = function([y, compile.In(x, update=new_x, value=12.0)], new_x)
    assert f(3) == 15
    f["x"] = None
    assert np.allclose(f(1), 4.8)
    assert np.allclose(f(np.asarray(2.2, dtype=config.floatX)), 7)


def test_autocast():
    backup_config = config.cast_policy
    # Call test functions for all possible values of `config.cast_policy`.
    for autocast_cfg in (
        "custom",
        # 'numpy', # Commented out until it is implemented properly.
        "numpy+floatX",
    ):
        config.cast_policy = autocast_cfg
        try:
            eval("_test_autocast_" + autocast_cfg.replace("+", "_"))()
        finally:
            config.cast_policy = backup_config


def _test_autocast_custom():
    # Called from `test_autocast`.
    assert config.cast_policy == "custom"
    orig_autocast = autocast_float.dtypes

    # Test that autocast_float_as sets the autocast dtype correctly
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    with autocast_float_as("float64"):
        assert autocast_float.dtypes == ("float64",)
    assert autocast_float.dtypes == orig_autocast

    # Test that we can set it back to something, and nest it
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
        with autocast_float_as("float64"):
            assert autocast_float.dtypes == ("float64",)
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float32"
        assert (fvector() + theano._asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + theano._asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.0).dtype == "float64"
        assert (fvector() + theano._asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + theano._asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32", "float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == theano.config.floatX
        assert (fvector() + 1.0).dtype == "float32"
        assert (dvector() + np.float32(1.1)).dtype == "float64"
        assert (dvector() + np.float64(1.1)).dtype == "float64"
        assert (dvector() + np.float(1.1)).dtype == "float64"
        assert (fvector() + np.float32(1.1)).dtype == "float32"
        assert (fvector() + np.float64(1.1)).dtype == "float64"
        assert (fvector() + np.float(1.1)).dtype == theano.config.floatX
        assert (lvector() + np.int64(1)).dtype == "int64"
        assert (lvector() + np.int32(1)).dtype == "int64"
        assert (lvector() + np.int16(1)).dtype == "int64"
        assert (lvector() + np.int8(1)).dtype == "int64"
        assert (ivector() + np.int8(1)).dtype == "int32"
        assert (wvector() + np.int8(1)).dtype == "int16"
        assert (bvector() + np.int8(1)).dtype == "int8"
        with autocast_float_as("float64"):
            assert (fvector() + 1.0).dtype == "float64"


def _test_autocast_numpy():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy"
    # Go through some typical scalar values.

    def ok(z):
        assert tt.constant(z).dtype == np.asarray(z).dtype

    for x in (
        [2 ** i for i in range(63)] + [0, 0, 1, 2 ** 63 - 1] + [0.0, 1.0, 1.1, 1.5]
    ):
        n_x = np.asarray(x)
        # Make sure the data type is the same as the one found by numpy.
        ok(x)
        ok(-x)
        ok(x - 1)
        ok(-x + 1)
        ok(n_x)


def _test_autocast_numpy_floatX():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy+floatX"
    backup_floatX = config.floatX

    def ok(z, floatX):
        if isinstance(z, float) and floatX == "float32" and not hasattr(z, "dtype"):
            # Special case where we use 'float32' instead of 'float64'.
            assert tt.constant(z).dtype == "float32"
        else:
            assert tt.constant(z).dtype == np.asarray(z).dtype

    try:
        # Test with various values of `config.floatX`.
        for floatX in ("float32", "float64"):
            config.floatX = floatX
            # Go through some typical scalar values.
            # We only consider 'int' and 'long' Python values that can fit
            # into int64, as that is the maximal integer type that Theano
            # supports, and that is the maximal type in Python indexing.
            for x in (
                [2 ** i - 1 for i in range(64)]
                + [0, 0, 1, 2 ** 63 - 1]
                + [0.0, 1.0, 1.1, 1.5]
            ):
                ok(x, floatX)
                ok(-x, floatX)
                ok(x - 1, floatX)
                ok(-x + 1, floatX)
                ok(np.asarray(x), floatX)
                ok(np.float64(x), floatX)
    finally:
        config.floatX = backup_floatX


class TestArithmeticCast:
    """Test output types of basic arithmeric operations (* / + - //).

    We only test the behavior for `config.cast_policy` set to either 'numpy' or
    'numpy+floatX': the 'custom' behavior is (at least partially) tested in
    `_test_autocast_custom`.

    """

    def test_arithmetic_cast(self):
        backup_config = config.cast_policy
        dtypes = get_numeric_types(with_complex=True)

        # Here:
        # scalar == scalar stored as a 0d array
        # array == 1d array
        # i_scalar == scalar type used internally by Theano
        def theano_scalar(dtype):
            return tt.scalar(dtype=str(dtype))

        def numpy_scalar(dtype):
            return np.array(1, dtype=dtype)

        def theano_array(dtype):
            return tt.vector(dtype=str(dtype))

        def numpy_array(dtype):
            return np.array([1], dtype=dtype)

        def theano_i_scalar(dtype):
            return theano.scalar.Scalar(str(dtype))()

        def numpy_i_scalar(dtype):
            return numpy_scalar(dtype)

        if config.int_division == "int":
            # Avoid deprecation warning during tests.
            warnings.filterwarnings(
                "ignore", message="Division of two integer", category=DeprecationWarning
            )
        try:
            for cfg in ("numpy+floatX",):  # Used to test 'numpy' as well.
                config.cast_policy = cfg
                for op in (
                    operator.add,
                    operator.sub,
                    operator.mul,
                    operator.truediv,
                    operator.floordiv,
                ):
                    for a_type in dtypes:
                        for b_type in dtypes:
                            # Note that we do not test division between
                            # integers if it is forbidden.
                            # Theano deals with integer division in its own
                            # special way (depending on `config.int_division`).
                            is_int_division = (
                                op is operator.truediv
                                and a_type in tt.discrete_dtypes
                                and b_type in tt.discrete_dtypes
                            )
                            # We will test all meaningful combinations of
                            # scalar and array operations.
                            for combo in (
                                ("scalar", "scalar"),
                                ("array", "array"),
                                ("scalar", "array"),
                                ("array", "scalar"),
                                ("i_scalar", "i_scalar"),
                            ):

                                theano_args = list(
                                    map(eval, ["theano_%s" % c for c in combo])
                                )
                                numpy_args = list(
                                    map(eval, ["numpy_%s" % c for c in combo])
                                )
                                try:
                                    theano_dtype = op(
                                        theano_args[0](a_type), theano_args[1](b_type)
                                    ).type.dtype
                                    # Should have crashed if it is an integer
                                    # division and `config.int_division` does
                                    # not allow it.
                                    assert not (
                                        is_int_division
                                        and config.int_division == "raise"
                                    )
                                except theano.scalar.IntegerDivisionError:
                                    assert (
                                        is_int_division
                                        and config.int_division == "raise"
                                    )
                                    # This is the expected behavior.
                                    continue
                                # For numpy we have a problem:
                                #   http://projects.scipy.org/numpy/ticket/1827
                                # As a result we only consider the highest data
                                # type that numpy may return.
                                numpy_dtypes = [
                                    op(
                                        numpy_args[0](a_type), numpy_args[1](b_type)
                                    ).dtype,
                                    op(
                                        numpy_args[1](b_type), numpy_args[0](a_type)
                                    ).dtype,
                                ]
                                numpy_dtype = theano.scalar.upcast(
                                    *list(map(str, numpy_dtypes))
                                )
                                if numpy_dtype == theano_dtype:
                                    # Same data type found, all is good!
                                    continue
                                if (
                                    cfg == "numpy+floatX"
                                    and config.floatX == "float32"
                                    and a_type != "float64"
                                    and b_type != "float64"
                                    and numpy_dtype == "float64"
                                ):
                                    # We should keep float32.
                                    assert theano_dtype == "float32"
                                    continue
                                if "array" in combo and "scalar" in combo:
                                    # For mixed scalar / array operations,
                                    # Theano may differ from numpy as it does
                                    # not try to prevent the scalar from
                                    # upcasting the array.
                                    array_type, scalar_type = (
                                        (a_type, b_type)[list(combo).index(arg)]
                                        for arg in ("array", "scalar")
                                    )
                                    up_type = theano.scalar.upcast(
                                        array_type, scalar_type
                                    )
                                    if (
                                        # The two data types are different.
                                        scalar_type != array_type
                                        and
                                        # The array type is not enough to hold
                                        # the scalar type as well.
                                        array_type != up_type
                                        and
                                        # Theano upcasted the result array.
                                        theano_dtype == up_type
                                        and
                                        # But Numpy kept its original type.
                                        array_type == numpy_dtype
                                    ):
                                        # Then we accept this difference in
                                        # behavior.
                                        continue
                                if is_int_division and config.int_division == "floatX":
                                    assert theano_dtype == config.floatX
                                    continue
                                if (
                                    cfg == "numpy+floatX"
                                    and a_type == "complex128"
                                    and (b_type == "float32" or b_type == "float16")
                                    and combo == ("scalar", "array")
                                    and theano_dtype == "complex128"
                                    and numpy_dtype == "complex64"
                                ):
                                    # In numpy 1.6.x adding a complex128 with
                                    # a float32 may result in a complex64. As
                                    # of 1.9.2. this is still the case so it is
                                    # probably by design
                                    pytest.skip("Known issue with" "numpy see #761")
                                # In any other situation: something wrong is
                                # going on!
                                raise AssertionError()
        finally:
            config.cast_policy = backup_config
            if config.int_division == "int":
                # Restore default deprecation warning behavior.
                warnings.filterwarnings(
                    "default",
                    message="Division of two integer",
                    category=DeprecationWarning,
                )


class TestLongTensor:
    def test_fit_int64(self):
        bitwidth = theano.configdefaults.python_int_bitwidth()
        for exponent in range(bitwidth):
            val = 2 ** exponent - 1
            scalar_ct = constant(val)

            assert scalar_ct.dtype in tt.int_dtypes, (
                exponent,
                val,
                scalar_ct.dtype,
            )
            assert scalar_ct.value == val

            vector_ct = constant([val, val])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert vector_ct.dtype == "int32"
            else:
                assert vector_ct.dtype == "int64"
            assert np.all(vector_ct.value == val)

            matrix_ct = constant([[val, val]])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert matrix_ct.dtype == "int32"
            else:
                assert matrix_ct.dtype == "int64"
            assert np.all(matrix_ct.value == val)

    def test_too_big(self):
        val = 2 ** 64
        # This fail for all NumPy version.
        with pytest.raises(Exception):
            constant(val)
        with pytest.raises(Exception):
            constant()[val, val]
        with pytest.raises(Exception):
            constant()[[val, val]]


class TestBroadcast:
    def test_broadcast_bigdim(self):
        def f():
            x = matrix()
            addbroadcast(x, 2)

        with pytest.raises(ValueError):
            f()

    def test_unbroadcast_addbroadcast(self):
        # test that the unbroadcast fct don't insert not needed broadcast
        # and fuse consecutive Rebroadcast op

        x = matrix()
        assert unbroadcast(x, 0) is x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is x
        assert unbroadcast(x, 0, 1) is x

        assert addbroadcast(x, 0) is not x
        assert addbroadcast(x, 1) is not x
        assert addbroadcast(x, 1, 0).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x, 0), 0) is x
        assert addbroadcast(unbroadcast(x, 0), 0) is not x
        x = row()
        assert unbroadcast(x, 0) is not x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is not x
        assert unbroadcast(x, 0, 1) is not x

        assert addbroadcast(x, 0) is x
        assert addbroadcast(x, 1).owner.inputs[0] is x
        assert addbroadcast(x, 1, 0).owner.inputs[0] is x
        assert addbroadcast(x, 0, 1).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x, 1), 1) is x
        assert addbroadcast(unbroadcast(x, 1), 1) is not x

        # The first broadcast is remove the broadcast, so the second
        # should not make one
        assert unbroadcast(unbroadcast(x, 0), 0).owner.inputs[0] is x

        # Test that consecutive Rebroadcast op are fused
        x = TensorType(dtype="float64", broadcastable=(True, True))()
        assert unbroadcast(unbroadcast(x, 1), 0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x, 1), 0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x, 0), 0) is x

    def test_patternbroadcast(self):
        # Test that patternbroadcast with an empty broadcasting pattern works
        x = scalar("x")
        m = tt.matrix("m")
        s = patternbroadcast(m, x.broadcastable)
        assert s is m
        x2 = patternbroadcast(x, x.broadcastable)
        assert x2 is x

    def test_infer_shape(self):
        x = matrix()
        y = addbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(np.zeros((1, 5), dtype=config.floatX)) == [1, 5]).all()
        topo = f.maker.fgraph.toposort()
        if theano.config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.MakeVector)

        x = matrix()
        y = unbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(np.zeros((2, 5), dtype=config.floatX)) == [2, 5]).all()
        topo = f.maker.fgraph.toposort()
        if theano.config.mode != "FAST_COMPILE":
            assert len(topo) == 3
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.Shape_i)
            assert isinstance(topo[2].op, opt.MakeVector)

        x = row()
        y = unbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(np.zeros((1, 5), dtype=config.floatX)) == [1, 5]).all()
        topo = f.maker.fgraph.toposort()
        if theano.config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.MakeVector)


def test_len():
    for shape_ in [(5,), (3, 4), (7, 4, 6)]:
        x = tt.tensor(dtype="floatX", broadcastable=(False,) * len(shape_))
        with pytest.raises(TypeError):
            len(x)


def test_divmod():
    # Confirm that divmod is equivalent to the python version.
    x, y = fscalars("xy")
    d, r = divmod(x, y)
    fn = gof.DualLinker().accept(gof.FunctionGraph([x, y], [d, r])).make_function()
    for a, b in (
        (0, 1),
        (1, 1),
        (0, -1),
        (1, -1),
        (-1, -1),
        (1, 2),
        (-1, 2),
        (1, -2),
        (-1, -2),
        (5, 3),
        (-5, 3),
        (5, -3),
        (-5, -3),
    ):
        d_v, r_v = fn(a, b)
        d_vp, r_vp = divmod(a, b)
        assert d_v == d_vp and r_v == r_vp, (a,)


def test_mod_compile():
    # This test generate an Elemwise of Composite as:
    #     Elemwise{
    #         Composite{
    #             Composite{
    #                 Composite{
    #                     Composite{mod,EQ},
    #                     Switch},
    #                 mul},
    #             add}}
    #
    # The c_code generated is not compiling as of 30 June 2010. I fix the
    # compilation in the same commit.
    x = tt.vector()
    y = tt.vector()
    out = tt.switch(tt.eq(3 % x.shape[0], 0), y, y[:-1])

    theano.function([x, y], out)


def test_unalign():
    if config.floatX == "float64":
        dtype = "b1,f8"
    else:
        dtype = "b1,f4"

    a = np.empty(10000, dtype=dtype)["f1"]
    b = np.empty(10000, dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    a[:] = rand(len(a))
    b[:] = rand(len(b))
    # out_numpy = 2 * a + 3 * b

    av, bv = tt.vectors("ab")
    f = theano.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()

    with pytest.raises(TypeError):
        f(a, b)

    a = np.empty((), dtype=dtype)["f1"]
    b = np.empty((), dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    # out_numpy = 2 * a + 3 * b

    av, bv = tt.scalars("ab")
    f = theano.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()
    with pytest.raises(TypeError):
        f(a, b)


def test_dimshuffle_duplicate():
    x = tt.vector()
    with pytest.raises(ValueError, match="may not appear twice"):
        tt.DimShuffle((False,), (0, 0))(x)


class TestGetScalarConstantValue:
    def test_get_scalar_constant_value(self):
        a = tt.stack([1, 2, 3])
        assert get_scalar_constant_value(a[0]) == 1
        assert get_scalar_constant_value(a[1]) == 2
        assert get_scalar_constant_value(a[2]) == 3

        b = tt.iscalar()
        a = tt.stack([b, 2, 3])
        with pytest.raises(tt.basic.NotScalarConstantError):
            get_scalar_constant_value(a[0])
        assert get_scalar_constant_value(a[1]) == 2
        assert get_scalar_constant_value(a[2]) == 3

        # For now get_scalar_constant_value goes through only MakeVector and Join of
        # scalars.
        v = tt.ivector()
        a = tt.stack([v, [2], [3]])
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(a[0])
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(a[1])
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(a[2])

        # Test the case SubTensor(Shape(v)) when the dimensions
        # is broadcastable.
        v = tt.row()
        assert get_scalar_constant_value(v.shape[0]) == 1

    def test_subtensor_of_constant(self):
        c = constant(rand(5))
        for i in range(c.value.shape[0]):
            assert get_scalar_constant_value(c[i]) == c.value[i]
        c = constant(rand(5, 5))
        for i in range(c.value.shape[0]):
            for j in range(c.value.shape[1]):
                assert get_scalar_constant_value(c[i, j]) == c.value[i, j]

    def test_numpy_array(self):
        # Regression test for crash when called on a numpy array.
        assert get_scalar_constant_value(np.array(3)) == 3
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(np.array([0, 1]))
        with pytest.raises(tt.EmptyConstantError):
            get_scalar_constant_value(np.array([]))

    def test_make_vector(self):
        mv = opt.make_vector(1, 2, 3)
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(mv)
        assert get_scalar_constant_value(mv[0]) == 1
        assert get_scalar_constant_value(mv[1]) == 2
        assert get_scalar_constant_value(mv[2]) == 3
        assert get_scalar_constant_value(mv[np.int32(0)]) == 1
        assert get_scalar_constant_value(mv[np.int64(1)]) == 2
        assert get_scalar_constant_value(mv[np.uint(2)]) == 3
        t = theano.scalar.Scalar("int64")
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(mv[t()])

    def test_shape_i(self):
        c = tt.constant(np.random.rand(3, 4))
        s = opt.Shape_i(0)(c)
        assert get_scalar_constant_value(s) == 3
        s = opt.Shape_i(1)(c)
        assert get_scalar_constant_value(s) == 4
        d = theano.shared(np.random.randn(1, 1), broadcastable=(True, True))
        f = tt.ScalarFromTensor()(opt.Shape_i(0)(d))
        assert get_scalar_constant_value(f) == 1

    def test_elemwise(self):
        # We test only for a few elemwise, the list of all supported
        # elemwise are in the fct.
        c = tt.constant(np.random.rand())
        s = c + 1
        assert np.allclose(get_scalar_constant_value(s), c.data + 1)
        s = c - 1
        assert np.allclose(get_scalar_constant_value(s), c.data - 1)
        s = c * 1.2
        assert np.allclose(get_scalar_constant_value(s), c.data * 1.2)
        s = c < 0.5
        assert np.allclose(get_scalar_constant_value(s), int(c.data < 0.5))
        s = tt.second(c, 0.4)
        assert np.allclose(get_scalar_constant_value(s), 0.4)

    def test_assert(self):
        # Make sure we still get the constant value if it is wrapped in
        # an Assert.
        c = tt.constant(2)
        x = tt.scalar()

        # condition is always True
        a = opt.Assert()(c, c > 1)
        assert get_scalar_constant_value(a) == 2

        with change_flags(compute_test_value="off"):
            # condition is always False
            a = opt.Assert()(c, c > 2)
            with pytest.raises(tt.NotScalarConstantError):
                get_scalar_constant_value(a)

        # condition is not constant
        a = opt.Assert()(c, c > x)
        with pytest.raises(tt.NotScalarConstantError):
            get_scalar_constant_value(a)

    def test_second(self):
        # Second should apply when the value is constant but not the shape
        c = tt.constant(np.random.rand())
        shp = tt.vector()
        s = tt.second(shp, c)
        assert get_scalar_constant_value(s) == c.data

    def test_copy(self):
        # Make sure we do not return the internal storage of a constant,
        # so we cannot change the value of a constant by mistake.
        c = tt.constant(3)
        d = extract_constant(c)
        d += 1
        e = extract_constant(c)
        assert e == 3, (c, d, e)


def test_complex_mod_failure():
    # Make sure % fails on complex numbers.
    x = vector(dtype="complex64")
    with pytest.raises(theano.scalar.ComplexError):
        x % 5


class TestSize:
    # Ensure the `size` attribute of tensors behaves as in numpy.
    def test_matrix(self):
        x = tt.matrix()
        y = np.zeros((5, 7), dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_vector(self):
        x = tt.vector()
        y = np.zeros(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_scalar(self):
        x = tt.scalar()
        y = np.array(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_shared(self):
        # NB: we also test higher order tensors at the same time.
        y = np.zeros((1, 2, 3, 4), dtype=config.floatX)
        x = theano.shared(y)
        assert y.size == function([], x.size)()


class TestDiag:
    # Test that tt.diag has the same behavior as np.diag.
    # np.diag has two behaviors:
    #
    # (1) when given a vector, it returns a matrix with that vector as the
    # diagonal.
    # (2) when given a matrix, returns a vector which is the diagonal of the
    # matrix.
    #
    # (1) and (2) are tested by test_alloc_diag and test_extract_diag
    # respectively.
    #
    # test_diag test makes sure that linalg.diag instantiates
    # the right op based on the dimension of the input.
    def setup_method(self):
        self.mode = None
        self.shared = tt._shared
        self.floatX = config.floatX
        self.type = tt.TensorType

    def test_diag(self):
        rng = np.random.RandomState(utt.fetch_seed())

        # test vector input
        x = tt.vector()
        g = diag(x)
        assert isinstance(g.owner.op, AllocDiag)
        f = theano.function([x], g)
        for shp in [5, 0, 1]:
            m = rng.rand(shp).astype(self.floatX)
            v = np.diag(m)
            r = f(m)
            # The right matrix is created
            assert (r == v).all()

        # Test matrix input
        xx = self.shared(rng.rand(3, 5))
        g = diag(xx)
        assert isinstance(g.owner.op, ExtractDiag)
        f = theano.function([], g)
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.rand(*shp).astype(self.floatX)
            xx.set_value(m)
            v = np.diag(m)
            r = f()
            # The right matrix is created
            assert (r == v).all()

        # Test scalar input
        xx = tt.scalar()
        with pytest.raises(ValueError):
            diag(xx)

    def test_infer_shape(self):
        rng = np.random.RandomState(utt.fetch_seed())

        x = tt.vector()
        g = diag(x)
        f = theano.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert np.sum([isinstance(node.op, AllocDiag) for node in topo]) == 0
        for shp in [5, 0, 1]:
            m = rng.rand(shp).astype(self.floatX)
            assert (f(m) == np.diag(m).shape).all()

        x = tt.matrix()
        g = diag(x)
        f = theano.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert np.sum([isinstance(node.op, ExtractDiag) for node in topo]) == 0
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.rand(*shp).astype(self.floatX)
            assert (f(m) == np.diag(m).shape).all()

    def test_diag_grad(self):
        rng = np.random.RandomState(utt.fetch_seed())
        x = rng.rand(5)
        tt.verify_grad(diag, [x], rng=rng)
        x = rng.rand(5, 3)
        tt.verify_grad(diag, [x], rng=rng)


class TestAllocDiag:
    def setup_method(self):
        self.alloc_diag = AllocDiag
        self.mode = theano.compile.mode.get_default_mode()

    def _generator(self):
        dims = 4
        shape = (5,) * dims
        xv = np.random.randn(*shape).astype(config.floatX)
        for d in range(1, dims + 1):
            # Create a TensorType of the same dimensions as
            # as the data we want to test.
            x = TensorType(dtype=config.floatX, broadcastable=(False,) * d)("x")

            # Make a slice of the test data that has the
            # dimensions we need by doing xv[0,...,0]
            # For example, for an array of shape (5,), we
            # need to do xv[0, 0, 0, 0].
            test_val = xv[((0,) * (dims - d))]
            yield x, test_val

    def test_alloc_diag_values(self):
        for x, test_val in self._generator():
            for offset, axis1, axis2 in [
                (0, 0, 1),
                (0, 1, 2),
                (1, 0, 1),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
                (-1, 0, 1),
                (-2, 0, 1),
                (-1, 1, 2),
            ]:
                # Test AllocDiag values
                if np.maximum(axis1, axis2) > len(test_val.shape):
                    continue
                adiag_op = self.alloc_diag(offset=offset, axis1=axis1, axis2=axis2)
                f = theano.function([x], adiag_op(x))
                # AllocDiag and extract the diagonal again
                # to check
                diag_arr = f(test_val)
                rediag = np.diagonal(diag_arr, offset=offset, axis1=axis1, axis2=axis2)
                assert np.all(rediag == test_val)

                # Test infer_shape
                f_shape = theano.function([x], adiag_op(x).shape, mode="FAST_RUN")

                theano.printing.debugprint(f_shape.maker.fgraph.outputs[0])
                output_shape = f_shape(test_val)
                assert not any(
                    isinstance(node.op, self.alloc_diag)
                    for node in f_shape.maker.fgraph.toposort()
                )
                rediag_shape = np.diagonal(
                    np.ones(output_shape), offset=offset, axis1=axis1, axis2=axis2
                ).shape
                assert np.all(rediag_shape == test_val.shape)

                diag_x = adiag_op(x)
                sum_diag_x = tt.sum(diag_x)
                grad_x = tt.grad(sum_diag_x, x)
                grad_diag_x = tt.grad(sum_diag_x, diag_x)
                f_grad_x = theano.function([x], grad_x, mode=self.mode)
                f_grad_diag_x = theano.function([x], grad_diag_x, mode=self.mode)
                grad_input = f_grad_x(test_val)
                grad_diag_input = f_grad_diag_x(test_val)
                true_grad_input = np.diagonal(
                    grad_diag_input, offset=offset, axis1=axis1, axis2=axis2
                )

                assert np.all(true_grad_input == grad_input)


class TestNumpyAssumptions:
    # Verify that some assumptions Theano makes on Numpy's behavior still hold.
    def test_ndarray_copy(self):
        # A copy or deepcopy of the ndarray type should not create a new object.
        #
        # This is because Theano makes some comparisons of the form:
        #     if type(x) is np.ndarray
        assert copy(np.ndarray) is np.ndarray
        assert deepcopy(np.ndarray) is np.ndarray

    def test_dtype_equality(self):
        # Ensure dtype string comparisons are consistent.
        #
        # Theano often uses string representations of dtypes (e.g. 'float32'). We
        # need to make sure that comparing the string representations is the same
        # as comparing the dtype objects themselves.
        dtypes = get_numeric_types(with_complex=True)
        # Perform all pairwise comparisons of dtypes, making sure comparing
        # their string representation yields the same result.
        for dtype1_idx, dtype1 in enumerate(dtypes):
            for dtype2 in dtypes[dtype1_idx + 1 :]:
                assert (dtype1 == dtype2) == (str(dtype1) == str(dtype2))


def test_transpose():
    x1 = tt.dvector("x1")
    x2 = tt.dmatrix("x2")
    x3 = tt.dtensor3("x3")

    x1v = np.arange(24)
    x2v = np.arange(24).reshape(2, 12)
    x3v = np.arange(24).reshape(2, 3, 4)

    f = theano.function(
        [x1, x2, x3],
        [
            tt.transpose(x1),
            tt.transpose(x2),
            tt.transpose(x3),
            x1.transpose(),
            x2.transpose(),
            x3.transpose(),
            x2.transpose(0, 1),
            x3.transpose((0, 2, 1)),
            tt.transpose(x2, [0, 1]),
            tt.transpose(x3, [0, 2, 1]),
        ],
    )

    t1, t2, t3, t1b, t2b, t3b, t2c, t3c, t2d, t3d = f(x1v, x2v, x3v)
    assert t1.shape == np.transpose(x1v).shape
    assert t2.shape == np.transpose(x2v).shape
    assert t3.shape == np.transpose(x3v).shape
    assert np.all(t1 == np.transpose(x1v))
    assert np.all(t2 == np.transpose(x2v))
    assert np.all(t3 == np.transpose(x3v))
    assert np.all(t1b == x1v.transpose())
    assert np.all(t2b == x2v.transpose())
    assert np.all(t3b == x3v.transpose())
    assert t2c.shape == (2, 12)
    assert t3c.shape == (2, 4, 3)
    assert np.all(t2c == x2v.transpose([0, 1]))
    assert np.all(t3c == x3v.transpose([0, 2, 1]))
    assert t2d.shape == (2, 12)
    assert t3d.shape == (2, 4, 3)
    assert np.all(t2d == np.transpose(x2v, [0, 1]))
    assert np.all(t3d == np.transpose(x3v, [0, 2, 1]))

    # Check that we create a name.
    assert tt.transpose(x1).name == "x1.T"
    assert tt.transpose(x2).name == "x2.T"
    assert tt.transpose(x3).name == "x3.T"
    assert tt.transpose(tt.dmatrix()).name is None


def test_stacklists():
    a, b, c, d = map(scalar, "abcd")
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2], [3, 4]]))

    X = stacklists([a, b, c, d])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (4,)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2, 3, 4]]))

    X = stacklists([[[a], [b]], [[c], [d]]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2, 1)

    a, b, c, d = [matrix(x) for x in "abcd"]
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    x = np.ones((4, 4), "float32")
    assert f(x, x, x, x).shape == (2, 2, 4, 4)


class TestInferShape(utt.InferShapeTester):
    def test_infer_shape(self):

        # Flatten
        atens3 = tensor3()
        atens3_val = rand(4, 5, 3)
        for outdim in (3, 2, 1):
            self._compile_and_check(
                [atens3],
                [flatten(atens3, outdim)],
                [atens3_val],
                Reshape,
                excluding=["local_useless_reshape"],
            )

        amat = matrix()
        amat_val = rand(4, 5)
        for outdim in (2, 1):
            self._compile_and_check(
                [amat],
                [flatten(amat, outdim)],
                [amat_val],
                Reshape,
                excluding=["local_useless_reshape"],
            )

        avec = vector()
        avec_val = rand(4)
        outdim = 1
        self._compile_and_check(
            [avec],
            [flatten(avec, outdim)],
            [avec_val],
            Reshape,
            excluding=["local_useless_reshape"],
        )

        # Eye
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 4, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 5, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [3, 5, 0], Eye
        )

        # Tri
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 4, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 5, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [3, 5, 0], Tri
        )

        # ExtractDiag
        atens3 = tensor3()
        atens3_val = rand(4, 5, 3)
        atens3_diag = ExtractDiag()(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(-1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 0, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 1, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 2, 0)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)

        # AllocDiag
        advec = dvector()
        advec_val = rand(4)
        self._compile_and_check([advec], [AllocDiag()(advec)], [advec_val], AllocDiag)

        # Shape
        # 'opt.Makevector' precludes optimizer from disentangling
        # elements of shape
        adtens = tensor3()
        adtens_val = rand(4, 5, 3)
        self._compile_and_check(
            [adtens], [Shape()(adtens)], [adtens_val], (opt.MakeVector, Shape)
        )

        # Dot

        # vec/vec
        advec = dvector()
        bdvec = dvector()
        advec_val = rand(4)
        bdvec_val = rand(4)
        self._compile_and_check(
            [advec, bdvec],
            [Dot()(advec, bdvec)],
            [advec_val, bdvec_val],
            (Dot, tt.blas.Dot22, tt.blas.Gemv, tt.blas_c.CGemv),
        )

        # mat/mat
        admat = dmatrix()
        bdmat = dmatrix()
        admat_val = rand(4, 5)
        bdmat_val = rand(5, 3)
        self._compile_and_check(
            [admat, bdmat],
            [Dot()(admat, bdmat)],
            [admat_val, bdmat_val],
            (Dot, tt.blas.Dot22),
        )

        # vec/mat
        bdmat_val = rand(4, 5)
        self._compile_and_check(
            [advec, bdmat],
            [Dot()(advec, bdmat)],
            [advec_val, bdmat_val],
            (Dot, tt.blas.Dot22, tt.blas.Gemv, tt.blas_c.CGemv),
        )

        # mat/vec
        admat_val = rand(5, 4)
        self._compile_and_check(
            [admat, bdvec],
            [Dot()(admat, bdvec)],
            [admat_val, bdvec_val],
            (Dot, tt.blas.Dot22, tt.blas.Gemv, tt.blas_c.CGemv),
        )

        # Split
        aivec = ivector()
        adtens_val = rand(4, 10, 3)
        aivec_val = [2, 5, 3]
        for aiscal_val in [1, -2]:
            self._compile_and_check(
                [adtens, aiscal, aivec],
                [Split(3)(adtens, aiscal, aivec)[0]],
                [adtens_val, aiscal_val, aivec_val],
                (Split),
            )

        # Join
        cdmat = dmatrix()
        admat_val = rand(1, 3)
        bdmat_val = rand(2, 3)
        cdmat_val = rand(4, 3)
        for aiscal_val in [0, -2]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

        admat_val = rand(4, 1)
        bdmat_val = rand(4, 3)
        cdmat_val = rand(4, 2)
        for aiscal_val in [-1, 1]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

        # PermuteRowElements
        abool = True
        rng = np.random.RandomState(utt.fetch_seed())
        advec_val = rand(5)
        aivec_val = rng.permutation(5).astype("int32")
        self._compile_and_check(
            [advec, aivec],
            [PermuteRowElements()(advec, aivec, abool)],
            [advec_val, aivec_val],
            PermuteRowElements,
        )

        admat_val = rand(3, 5)
        self._compile_and_check(
            [admat, aivec],
            [PermuteRowElements()(admat, aivec, abool)],
            [admat_val, aivec_val],
            PermuteRowElements,
        )

        adtens3 = dtensor3()
        adtens3_val = rand(3, 2, 5)
        self._compile_and_check(
            [adtens3, aivec],
            [PermuteRowElements()(adtens3, aivec, abool)],
            [adtens3_val, aivec_val],
            PermuteRowElements,
        )

        aimat = imatrix()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        aimat_val = np.vstack((perma, permb, permc))
        admat_val = rand(3, 5)
        self._compile_and_check(
            [admat, aimat],
            [PermuteRowElements()(admat, aimat, abool)],
            [admat_val, aimat_val],
            PermuteRowElements,
        )

        aitens3 = itensor3()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        bimat_val = np.vstack((perma, permb, permc))
        aitens3_val = np.empty((2, 3, 5), "int32")
        aitens3_val[0, ::, ::] = aimat_val
        aitens3_val[1, ::, ::] = bimat_val
        self._compile_and_check(
            [admat, aitens3],
            [PermuteRowElements()(admat, aitens3, abool)],
            [admat_val, aitens3_val],
            PermuteRowElements,
        )

        # ScalarFromTensor
        aiscal = iscalar()
        self._compile_and_check(
            [aiscal],
            [TensorFromScalar()(ScalarFromTensor()(aiscal))],
            [45],
            ScalarFromTensor,
            excluding=["local_tensor_scalar_tensor"],
        )

        # TensorFromScalar
        aiscal = scal.float64()

        self._compile_and_check(
            [aiscal], [TensorFromScalar()(aiscal)], [4.0], TensorFromScalar
        )

        # Alloc
        randint = np.random.randint
        adscal = dscalar()
        aiscal = lscalar()
        biscal = lscalar()
        ciscal = lscalar()
        discal = lscalar()
        adscal_val = rand()
        aiscal_val = randint(3, 6, size=())
        biscal_val = randint(3, 6, size=())
        ciscal_val = randint(3, 6, size=())
        discal_val = randint(3, 6, size=())
        self._compile_and_check(
            [adscal, aiscal, biscal, ciscal, discal],
            [Alloc()(adscal, aiscal, biscal, ciscal, discal)],
            [adscal_val, aiscal_val, biscal_val, ciscal_val, discal_val],
            Alloc,
        )

        # MaxAndArgmax,
        adtens3_val = rand(4, 5, 3)
        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, None), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 0), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 1), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 2), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, [0, 1, 2]), [adtens3_val], MaxAndArgmax
        )

        # ARange
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 5, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [2, 11, 4],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [-5, 1, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, -2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, 2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 0, 1],
            ARange,
        )

        # Mean
        adtens3_val = rand(3, 4, 5)
        aiscal_val = 2
        self._compile_and_check([adtens3], [Mean(None)(adtens3)], [adtens3_val], Mean)
        self._compile_and_check(
            [adtens3], [Mean(aiscal_val)(adtens3)], [adtens3_val], Mean
        )

        # Reshape
        # TODO: generalize infer_shape to account for tensor variable
        # (non-constant) input shape
        admat = dmatrix()
        aivec = ivector()
        ndim = 1
        admat_val = rand(3, 4)
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [12])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1])], [admat_val], Reshape
        )

        ndim = 2
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [4, 3])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [4, -1])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [3, -1])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1, 3])], [admat_val], Reshape
        )
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1, 4])], [admat_val], Reshape
        )

        # enable when infer_shape is generalized:
        # self._compile_and_check([admat, aivec],
        #                        [Reshape(ndim)(admat, aivec)],
        #                        [admat_val, [4, 3]], Reshape)
        #
        # self._compile_and_check([admat, aivec],
        #                        [Reshape(ndim)(admat, aivec)],
        #                        [admat_val, [4, -1]], Reshape)

        adtens4 = dtensor4()
        ndim = 4
        adtens4_val = rand(2, 4, 3, 5)
        self._compile_and_check(
            [adtens4], [Reshape(ndim)(adtens4, [1, -1, 10, 4])], [adtens4_val], Reshape
        )

        self._compile_and_check(
            [adtens4], [Reshape(ndim)(adtens4, [1, 3, 10, 4])], [adtens4_val], Reshape
        )

        # enable when infer_shape is generalized:
        # self._compile_and_check([adtens4, aivec],
        #                        [Reshape(ndim)(adtens4, aivec)],
        #                        [adtens4_val, [1, -1, 10, 4]], Reshape)
        #
        # self._compile_and_check([adtens4, aivec],
        #                        [Reshape(ndim)(adtens4, aivec)],
        #                        [adtens4_val, [1, 3, 10, 4]], Reshape)

        # Tile op is deprecated so the tile function doesn't use it
        # anymore, we'll test here the op directly
        advec = dvector()
        advec_val = rand(5)
        aivec_val = [3]
        ndim = 1
        self._compile_and_check(
            [advec], [Tile(ndim)(advec, aivec_val)], [advec_val], Tile
        )

        admat = dmatrix()
        admat_val = rand(2, 4)
        aivec_val = [2, 3]
        ndim = 2
        self._compile_and_check(
            [admat], [Tile(ndim)(admat, aivec_val)], [admat_val], Tile
        )

        adtens4 = dtensor4()
        adtens4_val = rand(2, 4, 3, 5)
        aivec_val = [2, 3, 1, 4]
        ndim = 4
        self._compile_and_check(
            [adtens4], [Tile(ndim)(adtens4, aivec_val)], [adtens4_val], Tile
        )


class TestTensorInstanceMethods:
    def setup_method(self):
        self.vars = matrices("X", "Y")
        self.vals = [m.astype(config.floatX) for m in [rand(2, 2), rand(2, 2)]]

    def test_argmin(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argmin().eval({X: x}), x.argmin())

    def test_argmax(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argmax().eval({X: x}), x.argmax())

    def test_argsort(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argsort().eval({X: x}), x.argsort())
        assert_array_equal(X.argsort(1).eval({X: x}), x.argsort(1))

    def test_clip(self):
        X, Y = self.vars
        x, y = self.vals
        # np.clip gives unexpected values when min > max,
        # so we have to make sure that min <= max in that test,
        # otherwise it randomly fails.
        Z = X.clip(Y - 0.5, Y + 0.5)
        z = x.clip(y - 0.5, y + 0.5)
        assert_array_equal(Z.eval({X: x, Y: y}), z)

    def test_dot(self):
        X, Y = self.vars
        x, y = self.vals
        # Use allclose comparison as a user reported on the mailing
        # list failure otherwise with array that print exactly the same.
        assert_allclose(x.dot(y), X.dot(Y).eval({X: x, Y: y}))
        Z = X.dot(Y)
        z = x.dot(y)
        assert_allclose(x.dot(z), X.dot(Z).eval({X: x, Z: z}))

    def test_real_imag(self):
        X, Y = self.vars
        x, y = self.vals
        Z = X + Y * 1j
        z = x + y * 1j
        assert_array_equal(Z.real.eval({Z: z}), x)
        assert_array_equal(Z.imag.eval({Z: z}), y)

    def test_conj(self):
        X, Y = self.vars
        x, y = self.vals
        Z = X + Y * 1j
        z = x + y * 1j
        assert_array_equal(Z.conj().eval({Z: z}), z.conj())
        assert_array_equal(Z.conjugate().eval({Z: z}), z.conj())

    def test_round(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.round().eval({X: x}), x.round())

    def test_std(self):
        X, _ = self.vars
        x, _ = self.vals
        # std() is implemented as theano tree and does not pass its
        # args directly to numpy. This sometimes results in small
        # difference, so we use allclose test.
        assert_allclose(X.std().eval({X: x}), x.std())

    def test_repeat(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.repeat(2).eval({X: x}), x.repeat(2))

    def test_trace(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.trace().eval({X: x}), x.trace())

    def test_ravel(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.ravel().eval({X: x}), x.ravel())

    def test_diagonal(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.diagonal().eval({X: x}), x.diagonal())
        assert_array_equal(X.diagonal(1).eval({X: x}), x.diagonal(1))
        assert_array_equal(X.diagonal(-1).eval({X: x}), x.diagonal(-1))
        for offset, axis1, axis2 in [(1, 0, 1), (-1, 0, 1), (0, 1, 0), (-2, 1, 0)]:
            assert_array_equal(
                X.diagonal(offset, axis1, axis2).eval({X: x}),
                x.diagonal(offset, axis1, axis2),
            )

    def test_take(self):
        X, _ = self.vars
        x, _ = self.vals
        indices = [1, 0, 3]
        assert_array_equal(X.take(indices).eval({X: x}), x.take(indices))
        indices = [1, 0, 1]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        indices = np.array([-10, 5, 12], dtype="int32")
        assert_array_equal(
            X.take(indices, 1, mode="wrap").eval({X: x}),
            x.take(indices, 1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="wrap").eval({X: x}),
            x.take(indices, -1, mode="wrap"),
        )
        assert_array_equal(
            X.take(indices, 1, mode="clip").eval({X: x}),
            x.take(indices, 1, mode="clip"),
        )
        assert_array_equal(
            X.take(indices, -1, mode="clip").eval({X: x}),
            x.take(indices, -1, mode="clip"),
        )
        # Test error handling
        with pytest.raises(IndexError):
            X.take(indices).eval({X: x})
        with pytest.raises(IndexError):
            (2 * X.take(indices)).eval({X: x})
        with pytest.raises(TypeError):
            X.take([0.0])
        indices = [[1, 0, 1], [0, 1, 1]]
        assert_array_equal(X.take(indices, 1).eval({X: x}), x.take(indices, 1))
        # Test equivalent advanced indexing
        assert_array_equal(X[:, indices].eval({X: x}), x[:, indices])

    def test_cumsum(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.cumsum().eval({X: x}), x.cumsum())

    def test_cumprod(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.cumprod().eval({X: x}), x.cumprod())


def test_norm():
    x = tt.vector("x")
    n = x.norm(2)
    f = theano.function([x], n)
    assert np.allclose(f([1, 1]), np.sqrt(2))


class TestCov:
    def test_core(self):
        x = tt.matrix("x")
        c = tt.cov(x)
        f = theano.function([x], c)

        # basic cov function
        data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data))

        data = np.asarray(np.random.rand(5, 3), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data))

        data = np.asarray(np.random.rand(10, 10), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data))

        data = np.asarray(np.random.rand(2, 2), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data))

        data = np.asarray(np.random.rand(1, 2), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data))

    def test_rowvar(self):
        for rowvar in [True, False]:
            x = tt.matrix("x")
            c = tt.cov(x, rowvar=rowvar)
            f = theano.function([x], c)

            data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, rowvar=rowvar))

            data = np.asarray(np.random.rand(5, 3), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, rowvar=rowvar))

            data = np.asarray(np.random.rand(10, 10), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, rowvar=rowvar))

            data = np.asarray(np.random.rand(2, 2), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, rowvar=rowvar))

        # check when variables are along the first axis
        x = tt.matrix("x")
        c = tt.cov(x, rowvar=False)
        f = theano.function([x], c)
        data = np.asarray(np.random.rand(2, 1), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data, rowvar=False))

    def test_y(self):
        # test y
        x = tt.matrix("x")
        y = tt.matrix("y")
        c = tt.cov(x, y=y)
        f = theano.function([x, y], c)

        data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        y = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        assert np.allclose(f(data, y), np.cov(data, y=y))

        data = np.asarray(np.random.rand(5, 3), dtype=config.floatX)
        y = np.asarray(np.random.rand(5, 3), dtype=config.floatX)
        assert np.allclose(f(data, y), np.cov(data, y=y))

        data = np.asarray(np.random.rand(10, 10), dtype=config.floatX)
        y = np.asarray(np.random.rand(10, 10), dtype=config.floatX)
        assert np.allclose(f(data, y), np.cov(data, y=y))

        data = np.asarray(np.random.rand(2, 2), dtype=config.floatX)
        y = np.asarray(np.random.rand(2, 2), dtype=config.floatX)
        assert np.allclose(f(data, y), np.cov(data, y=y))

    def test_ddof(self):

        for ddof in range(0, 5):
            x = tt.matrix("x")
            c = tt.cov(x, ddof=ddof)
            f = theano.function([x], c)

            data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, ddof=ddof))

    def test_bias(self):

        for bias in [True, False]:
            x = tt.matrix("x")
            c = tt.cov(x, bias=bias)
            f = theano.function([x], c)

            data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
            assert np.allclose(f(data), np.cov(data, bias=bias))

        for ddof in range(0, 5):
            for bias in [True, False]:
                x = tt.matrix("x")
                c = tt.cov(x, ddof=ddof, bias=bias)
                f = theano.function([x], c)

                data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
                assert np.allclose(f(data), np.cov(data, ddof=ddof, bias=bias))


class TestPtp:
    def test_scalar(self):
        # Should return 0 for all scalar
        x = scalar("x")
        p = ptp(x)
        f = theano.function([x], p)

        y = np.asarray(rand() * 2000 - 1000, dtype=config.floatX)
        result = f(y)
        numpyResult = np.ptp(y)

        assert np.array_equal(result, numpyResult)

    def test_vector(self):

        x = vector("x")
        p = ptp(x, 0)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100])
        result = f(y)
        numpyResult = np.ptp(y, 0)

        assert np.array_equal(result, numpyResult)

    def test_matrix_first_axis(self):

        x = matrix("x")
        p = ptp(x, 1)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100, 100])
        result = f(y)
        numpyResult = np.ptp(y, 1)

        assert np.array_equal(result, numpyResult)

    def test_matrix_second_axis(self):
        x = matrix("x")
        p = ptp(x, 0)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100, 100])
        result = f(y)
        numpyResult = np.ptp(y, 0)

        assert np.array_equal(result, numpyResult)

    def test_matrix_neg_axis(self):
        x = matrix("x")
        p = ptp(x, -1)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100, 100])
        result = f(y)
        numpyResult = np.ptp(y, -1)

        assert np.array_equal(result, numpyResult)

    def test_matrix_no_axis(self):
        x = matrix("x")
        p = ptp(x)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100, 100])
        result = f(y)
        numpyResult = np.ptp(y)

        assert np.array_equal(result, numpyResult)

    def test_interface(self):
        x = matrix("x")
        p = x.ptp(1)
        f = theano.function([x], p)

        y = rand_ranged(-1000, 1000, [100, 100])
        result = f(y)
        numpyResult = np.ptp(y, 1)

        assert np.array_equal(result, numpyResult)


class TestSwapaxes:
    def test_no_dimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes(2, 0, 1)

    def test_unidimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes([2, 1], 0, 1)

    def test_not_enough_dimension(self):
        with pytest.raises(IndexError):
            swapaxes([[2, 1], [3, 4]], 3, 4)

    def test_doubleswap(self):
        y = matrix()
        n = swapaxes(y, 0, 1)
        f = function([y], n)
        testMatrix = [[2, 1], [3, 4]]
        assert np.array_equal(testMatrix, f(f(testMatrix)))

    def test_interface(self):
        x = tt.matrix()
        x.swapaxes(0, 1)

    def test_numpy_compare(self):
        rng = np.random.RandomState(utt.fetch_seed())
        A = tt.matrix("A", dtype=theano.config.floatX)
        Q = swapaxes(A, 0, 1)
        fn = function([A], [Q])
        a = rng.rand(4, 4).astype(theano.config.floatX)

        n_s = np.swapaxes(a, 0, 1)
        t_s = fn(a)
        assert np.allclose(n_s, t_s)


class TestPower:
    def test_numpy_compare(self):
        rng = np.random.RandomState(utt.fetch_seed())
        A = tt.matrix("A", dtype=theano.config.floatX)
        Q = power(A, 3)
        fn = function([A], [Q])
        a = rng.rand(4, 4).astype(theano.config.floatX)

        n_p = np.power(a, 3)
        t_p = fn(a)
        assert np.allclose(n_p, t_p)

    def test_multiple_power(self):
        x = tt.vector()
        y = [1, 2, 3]
        z = power(x, y)
        f = function([x], z)
        assert np.allclose(f([1, 2, 3]), [1, 4, 27])

    def test_wrong_shape(self):
        x = tt.vector()
        y = [1, 2, 3]
        z = power(x, y)
        f = function([x], z)
        with pytest.raises(ValueError):
            f([1, 2, 3, 4])


class TestChoose(utt.InferShapeTester):
    op = staticmethod(choose)
    op_class = Choose
    modes = ["raise", "wrap", "clip"]

    def test_numpy_compare(self):

        a = tt.vector(dtype="int32")
        b = tt.matrix(dtype="float32")

        A = np.random.randint(0, 4, 4).astype("int32")
        B = np.asarray(np.random.rand(4, 4), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_method(self):
        a = tt.vector(dtype="int32")
        b = tt.matrix(dtype="float32")

        A = np.random.randint(0, 4, 4).astype("int32")
        B = np.asarray(np.random.rand(4, 4), dtype="float32")

        for m in self.modes:
            f = function([a, b], a.choose(b, mode=m))
            t_c = f(A, B)
            n_c = A.choose(B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_broadcasted(self):
        a = tt.scalar(dtype="int32")
        b = tt.matrix(dtype="float32")

        # Test when a is broadcastable
        A = 3
        B = np.asarray(np.random.rand(4, 4), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

        # Test when the result should be broadcastable
        b = tt.col(dtype="float32")
        B = np.asarray(np.random.rand(4, 1), dtype="float32")
        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            assert choose(a, b, mode=m).broadcastable[0]
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_dtype_error(self):
        a = tt.scalar(dtype="float32")
        b = tt.matrix(dtype="float32")

        with pytest.raises(TypeError):
            choose(a, b)

    def test_numpy_compare_tuple(self):

        a = tt.tensor3(dtype="int32")
        b = tt.tensor3(dtype="float32")
        c = tt.tensor3(dtype="float32")

        A = np.random.randint(0, 2, (2, 1, 1)).astype("int32")
        B = np.asarray(np.random.rand(1, 6, 1), dtype="float32")
        C = np.asarray(np.random.rand(1, 1, 5), dtype="float32")

        for m in self.modes:
            f = function([a, b, c], choose(a, (b, c), mode=m))
            t_c = f(A, B, C)
            n_c = np.choose(A, (B, C), mode=m)
            assert np.allclose(t_c, n_c)

    def test_infer_shape(self):
        for shp1, shp2 in [
            ((5, 4), (7, 4)),
            ((1, 4), (7, 4)),
            ((5, 1), (7, 4)),
            ((5, 4), (1, 4)),
            ((5, 4), (7, 1)),
            ((5, 4), (4,)),
            ((1, 4), (4,)),
            ((5, 1), (4,)),
            ((5, 4), (1,)),
            ((4,), (5, 4)),
            ((1,), (5, 4)),
            ((4,), (1, 4)),
            ((4,), (3, 1)),
            ((4,), (4,)),
            ((1,), (4,)),
            ((4,), (1,)),
            ((1,), (1,)),
        ]:
            a = tt.tensor(dtype="int32", broadcastable=[n == 1 for n in shp1])
            c = tt.tensor(dtype="float32", broadcastable=[n == 1 for n in shp2])
            A = np.asarray(np.random.rand(*shp1) * shp2[0], dtype="int32")
            C = np.asarray(np.random.rand(*shp2) * shp2[0], dtype="float32")
            self._compile_and_check(
                [a, c],  # theano.function inputs
                [self.op(a, c)],  # theano.function outputs
                # Always use not square matrix!
                # inputs data
                [A, C],
                # Op that should be removed from the graph.
                self.op_class,
            )

    @pytest.mark.skip(reason="Not implemented")
    def test_infer_shape_tuple(self):

        a = tt.tensor3(dtype="int32")
        b = tt.tensor3(dtype="int32")
        c = tt.tensor3(dtype="int32")

        A = np.asarray([1, 0], dtype="int32").reshape((2, 1, 1))
        B = np.asarray(np.random.rand(1, 4, 1), dtype="int32")
        C = np.asarray(np.random.rand(1, 1, 7), dtype="int32")

        f = function([a, b, c], choose(a, (b, c)))
        shape = (2, 4, 7)
        assert np.allclose(f(A, B, C).shape, shape)

        self._compile_and_check(
            [a, b, c],  # theano.function inputs
            [self.op(a, (b, c))],  # theano.function outputs
            # Always use not square matrix!
            # inputs data
            [A, B, C],
            # Op that should be removed from the graph.
            self.op_class,
        )


def test_allocempty():
    # Test that we allocated correctly
    f = theano.function([], AllocEmpty("float32")(2, 3))
    assert len(f.maker.fgraph.apply_nodes) == 1
    out = f()

    assert out.shape == (2, 3)
    assert out.dtype == "float32"


def test_symbolic_slice():
    x = tt.tensor4("x")
    a, b = x.shape[:2]
    output = a.eval({x: np.zeros((5, 4, 3, 2), dtype=theano.config.floatX)})
    assert output == np.array(5)

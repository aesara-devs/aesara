import numpy as np
import pytest

import aesara
from aesara import function
from aesara.compile.io import In
from aesara.misc.safe_asarray import _asarray
from aesara.tensor.basic import (
    _convert_to_complex64,
    _convert_to_complex128,
    _convert_to_float32,
    _convert_to_float64,
    _convert_to_int8,
    _convert_to_int16,
    _convert_to_int32,
    _convert_to_int64,
    cast,
)
from aesara.tensor.type import (
    TensorType,
    bvector,
    dmatrix,
    dvector,
    fvector,
    ivector,
    zmatrix,
)


class TestCasting:
    @pytest.mark.parametrize(
        "op_fn", [_convert_to_int32, _convert_to_float32, _convert_to_float64]
    )
    @pytest.mark.parametrize("type_fn", [bvector, ivector, fvector, dvector])
    def test_0(self, op_fn, type_fn):
        x = type_fn()
        f = function([x], op_fn(x))

        xval = _asarray(np.random.random(10) * 10, dtype=type_fn.dtype)
        yval = f(xval)
        assert str(yval.dtype) == op_fn.scalar_op.output_types_preference.spec[0].dtype

    def test_illegal(self):
        x = zmatrix()
        with pytest.raises(TypeError):
            function([x], cast(x, "float64"))(np.ones((2, 3), dtype="complex128"))

    @pytest.mark.parametrize(
        "type1",
        [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        ],
    )
    @pytest.mark.parametrize(
        "type2, converter",
        zip(
            ["int8", "int16", "int32", "int64", "float32", "float64"],
            [
                _convert_to_int8,
                _convert_to_int16,
                _convert_to_int32,
                _convert_to_int64,
                _convert_to_float32,
                _convert_to_float64,
            ],
        ),
    )
    def test_basic(self, type1, type2, converter):
        x = TensorType(dtype=type1, shape=(None,))()
        y = converter(x)
        f = function([In(x, strict=True)], y)
        a = np.arange(10, dtype=type1)
        b = f(a)
        assert np.array_equal(b, np.arange(10, dtype=type2))

    def test_convert_to_complex(self):
        val64 = np.ones(3, dtype="complex64") + 0.5j
        val128 = np.ones(3, dtype="complex128") + 0.5j

        vec64 = TensorType("complex64", shape=(None,))()
        vec128 = TensorType("complex128", shape=(None,))()

        f = function([vec64], _convert_to_complex128(vec64))
        # we need to compare with the same type.
        assert vec64.type.values_eq_approx(val128, f(val64))

        f = function([vec128], _convert_to_complex128(vec128))
        assert vec64.type.values_eq_approx(val128, f(val128))

        f = function([vec64], _convert_to_complex64(vec64))
        assert vec64.type.values_eq_approx(val64, f(val64))

        f = function([vec128], _convert_to_complex64(vec128))
        assert vec128.type.values_eq_approx(val64, f(val128))

        # upcasting to complex128
        for t in ["int8", "int16", "int32", "int64", "float32", "float64"]:
            a = aesara.shared(np.ones(3, dtype=t))
            b = aesara.shared(np.ones(3, dtype="complex128"))
            f = function([], _convert_to_complex128(a))
            assert a.type.values_eq_approx(b.get_value(), f())

        # upcasting to complex64
        for t in ["int8", "int16", "int32", "int64", "float32"]:
            a = aesara.shared(np.ones(3, dtype=t))
            b = aesara.shared(np.ones(3, dtype="complex64"))
            f = function([], _convert_to_complex64(a))
            assert a.type.values_eq_approx(b.get_value(), f())

        # downcast to complex64
        for t in ["float64"]:
            a = aesara.shared(np.ones(3, dtype=t))
            b = aesara.shared(np.ones(3, dtype="complex64"))
            f = function([], _convert_to_complex64(a))
            assert a.type.values_eq_approx(b.get_value(), f())

    def test_bug_complext_10_august_09(self):
        v0 = dmatrix()
        v1 = _convert_to_complex128(v0)

        f = function([v0], v1)
        i = np.zeros((2, 2))
        assert np.array_equal(f(i), i)

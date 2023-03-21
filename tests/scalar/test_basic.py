import numpy as np
import pytest

import aesara
import aesara.tensor as at
import tests.unittest_tools as utt
from aesara.compile.mode import Mode
from aesara.graph.fg import FunctionGraph
from aesara.link.c.basic import DualLinker
from aesara.scalar.basic import (
    ComplexError,
    Composite,
    InRange,
    ScalarType,
    add,
    and_,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cast,
    complex64,
    constant,
    cos,
    cosh,
    deg2rad,
    eq,
    exp,
    exp2,
    expm1,
    float16,
    float32,
    floats,
    int8,
    int32,
    ints,
    invert,
    log,
    log1p,
    log2,
    log10,
    mean,
    mul,
    neq,
    rad2deg,
    reciprocal,
    sin,
    sinh,
    sqrt,
    switch,
    tan,
    tanh,
    true_divide,
    uint8,
)
from aesara.tensor.type import fscalar, imatrix, iscalar, matrix
from tests.link.test_link import make_function


def test_mul_add_true():
    x, y, z = floats("xyz")
    e = mul(add(x, y), true_divide(x, y))
    g = FunctionGraph([x, y], [e])
    fn = make_function(DualLinker().accept(g))
    assert fn(1.0, 2.0) == 1.5


class TestComposite:
    def test_composite_clone_float32(self):
        def has_f16(comp):
            if any(v.type == float16 for v in comp.fgraph.variables):
                return True
            return False

        w = int8()
        x = float16()
        y = float32()
        cz = Composite([x, y], [tanh(x + cast(y, "float16"))])
        c = Composite(
            [w, x, y],
            [
                cz(x, y)
                - cz(x, y) ** 2
                + cast(x, "int16")
                + cast(x, "float32")
                + cast(w, "float16")
                - constant(np.float16(1.0))
            ],
        )
        assert has_f16(c)
        nc = c.clone_float32()
        assert not has_f16(nc)

        v = uint8()
        w = float16()
        x = float16()
        y = float16()
        z = float16()

        c = Composite([v, w, x, y, z], [switch(v, mul(w, x, y), z)])

        assert has_f16(c)
        nc = c.clone_float32()
        assert not has_f16(nc)

    def test_straightforward(self):
        x, y, z = floats("xyz")
        e = mul(add(x, y), true_divide(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = FunctionGraph([x, y], [c.out])
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0) == 1.5

    def test_flatten(self):
        # Test that we flatten multiple Composite.
        x, y, z = floats("xyz")
        C = Composite([x, y], [x + y])
        CC = Composite([x, y], [C(x * y, y)])
        assert not isinstance(CC.outputs[0].owner.op, Composite)

        # Test with multiple outputs
        CC = Composite([x, y, z], [C(x * y, y), C(x * z, y)])
        # We don't flatten that case.
        assert isinstance(CC.outputs[0].owner.op, Composite)

    def test_with_constants(self):
        x, y, z = floats("xyz")
        e = mul(add(70.0, y), true_divide(x, y))
        comp_op = Composite([x, y], [e])
        comp_node = comp_op.make_node(x, y)

        c_code = comp_node.op.c_code(comp_node, "dummy", ["x", "y"], ["z"], dict(id=0))
        assert "70.0" in c_code

        # Make sure caching of the c_code template works
        assert hasattr(comp_node.op, "_c_code")

        g = FunctionGraph([x, y], [comp_node.out])
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0) == 36.0

    def test_many_outputs(self):
        x, y, z = floats("xyz")
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        C = Composite([x, y, z], [e0, e1, e2])
        c = C.make_node(x, y, z)
        # print c.c_code(['x', 'y', 'z'], ['out0', 'out1', 'out2'], dict(id = 0))
        g = FunctionGraph([x, y, z], c.outputs)
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0, 3.0) == [6.0, 7.0, 0.5]

    def test_composite_printing(self):
        x, y, z = floats("xyz")
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        e3 = x // 5
        e4 = -x
        e5 = x - y
        e6 = x**y + (-z)
        e7 = x % 3
        C = Composite([x, y, z], [e0, e1, e2, e3, e4, e5, e6, e7])
        c = C.make_node(x, y, z)
        g = FunctionGraph([x, y, z], c.outputs)
        make_function(DualLinker().accept(g))

        assert str(g) == (
            "FunctionGraph(*1 -> Composite{((i0 + i1) + i2),"
            " (i0 + (i1 * i2)), (i0 / i1), "
            "(i0 // 5), "
            "(-i0), (i0 - i1), ((i0 ** i1) + (-i2)),"
            " (i0 % 3)}(x, y, z), "
            "*1::1, *1::2, *1::3, *1::4, *1::5, *1::6, *1::7)"
        )

    def test_non_scalar_error(self):
        x = float32("x")
        comp_op = Composite([x], [(at.zeros((2,)) + x).sum()])

        with pytest.raises(TypeError, match=".*exclusively.*ScalarOp.*"):
            comp_op.fgraph

    def test_multi_out_perform(self):
        from aesara.graph.basic import Apply
        from aesara.scalar.basic import ScalarOp

        class MultiOutOp(ScalarOp):
            def make_node(self, x):
                return Apply(self, [x], [x.type(), x.type()])

            def perform(self, node, inputs, outputs):
                outputs[1][0] = outputs[0][0] = inputs[0]

            def c_code(self, *args):
                return "dummy"

        x = float32("x")
        comp_op = Composite([x], MultiOutOp()(x))

        y, z = comp_op(x)

        fn = aesara.function([x], [y, z], mode=Mode("py", None))

        assert fn(1.0) == [1.0, 1.0]


class TestLogical:
    def test_gt(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x > y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a > b)

    def test_lt(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x < y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a < b)

    def test_le(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x <= y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a <= b)

    def test_ge(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x >= y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a >= b)

    def test_eq(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [eq(x, y)])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a == b)

    def test_neq(self):
        x, y, z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [neq(x, y)])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a != b)

    def test_or(self):
        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x | y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a | b), (a, b)

    def test_xor(self):
        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x ^ y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a ^ b), (a, b)

    def test_and(self):
        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [and_(x, y)])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a & b), (a, b)

        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x & y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a & b), (a, b)

    def test_not(self):
        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [invert(x)])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == ~a, (a,)

        x, y, z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [~x])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == ~a, (a,)


class TestUpgradeToFloat:
    # Test for Ops whose output has to be floating point, even when all
    # inputs are ints.
    # In particular, when the inputs are int8, the output should be
    # at least float32, not float16.

    unary_ops_vals = [
        (reciprocal, list(range(-127, 0)) + list(range(1, 127))),
        (sqrt, list(range(0, 128))),
        (log, list(range(1, 128))),
        (log2, list(range(1, 128))),
        (log10, list(range(1, 128))),
        (log1p, list(range(0, 128))),
        (exp, list(range(-127, 89))),
        (exp2, list(range(-127, 89))),
        (expm1, list(range(-127, 89))),
        (deg2rad, list(range(-127, 128))),
        (rad2deg, list(range(-127, 128))),
        (cos, list(range(-127, 128))),
        (arccos, list(range(-1, 2))),
        (cosh, list(range(-89, 90))),
        (arccosh, list(range(1, 128))),
        (sin, list(range(-127, 128))),
        (arcsin, list(range(-1, 2))),
        (sinh, list(range(-89, 90))),
        (arcsinh, list(range(-127, 128))),
        (tan, list(range(-3, 4))),
        (arctan, list(range(-127, 128))),
        (tanh, list(range(-127, 128))),
        (arctanh, [0]),
    ]

    binary_ops_vals = [(arctan2, list(range(-127, 128)), list(range(-127, 128)))]

    @staticmethod
    def _test_unary(unary_op, x_range):
        xi = int8("xi")
        xf = float32("xf")

        ei = unary_op(xi)
        fi = aesara.function([xi], ei)

        ef = unary_op(xf)
        ff = aesara.function([xf], ef)

        for x_val in x_range:
            outi = fi(x_val)
            outf = ff(x_val)

            assert outi.dtype == outf.dtype, "incorrect dtype"
            assert np.allclose(outi, outf), "insufficient precision"

    @staticmethod
    def _test_binary(binary_op, x_range, y_range):
        xi = int8("xi")
        yi = int8("yi")
        xf = float32("xf")
        yf = float32("yf")

        ei = binary_op(xi, yi)
        fi = aesara.function([xi, yi], ei)

        ef = binary_op(xf, yf)
        ff = aesara.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                assert outi.dtype == outf.dtype, "incorrect dtype"
                assert np.allclose(outi, outf), "insufficient precision"

    def test_true_divide(self):
        # true_divide's upcast policy is not exactly "upgrade_to_float",
        # so the test is a little bit different
        x_range = list(range(-127, 128))
        y_range = list(range(-127, 0)) + list(range(1, 127))

        xi = int8("xi")
        yi = int8("yi")
        xf = ScalarType(aesara.config.floatX)("xf")
        yf = ScalarType(aesara.config.floatX)("yf")

        ei = true_divide(xi, yi)
        fi = aesara.function([xi, yi], ei)

        ef = true_divide(xf, yf)
        ff = aesara.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                assert outi.dtype == outf.dtype, "incorrect dtype"
                assert np.allclose(outi, outf), "insufficient precision"

    def test_unary(self):
        # Automatically define all individual unary tests
        for unary_op, x_range in self.unary_ops_vals:
            self._test_unary(unary_op, x_range)

    def test_binary(self):
        # Automatically define all individual binary tests
        for binary_op, x_range, y_range in self.binary_ops_vals:
            self._test_binary(binary_op, x_range, y_range)


def test_mod_complex_fail():
    # Make sure % fails on complex numbers.
    x = complex64()
    y = int32()
    with pytest.raises(ComplexError):
        x % y


def test_grad_gt():
    x = float32(name="x")
    y = float32(name="y")
    z = x > y
    g = aesara.gradient.grad(z, y)
    assert g.eval({y: 1.0}) == 0.0


def test_grad_switch():
    # This is a code snippet from the mailing list
    # It caused an assert to be raised due to the
    # switch op's grad method not handling integer
    # inputs correctly

    x = matrix()
    c = matrix()

    s = aesara.tensor.switch(c, x, 0)
    l = s.sum()

    aesara.gradient.grad(l, x)


def test_grad_identity():
    # Check that the grad method of Identity correctly handles int dytpes
    x = imatrix("x")
    # tensor_copy is Elemwise{Identity}
    y = aesara.tensor.tensor_copy(x)
    l = y.sum(dtype=aesara.config.floatX)
    aesara.gradient.grad(l, x)


def test_grad_inrange():
    for bound_definition in [(True, True), (False, False)]:
        # Instantiate op, and then take the gradient
        op = InRange(*bound_definition)
        x = fscalar("x")
        low = fscalar("low")
        high = fscalar("high")
        out = op(x, low, high)
        gx, glow, ghigh = aesara.gradient.grad(out, [x, low, high])

        # We look if the gradient are equal to zero
        # if x is lower than the lower bound,
        # equal to the lower bound, between lower and higher bound,
        # equal to the higher bound and higher than the higher
        # bound.
        # Mathematically we should have an infinite gradient when
        # x is equal to the lower or higher bound but in that case
        # Aesara defines the gradient to be zero for stability.
        f = aesara.function([x, low, high], [gx, glow, ghigh])
        utt.assert_allclose(f(0, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(1, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(2, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(5, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(7, 1, 5), [0, 0, 0])


def test_grad_abs():
    a = fscalar("a")
    b = 0.5 * (a + aesara.tensor.abs(a))
    c = aesara.grad(b, a)
    f = aesara.function([a], c, mode=Mode(optimizer=None))
    # Currently Aesara return 0.5, but it isn't sure it won't change
    # in the futur.
    ret = f(0.0)
    assert ret == 0.5, ret


def test_constant():
    c = constant(2, name="a")
    assert c.name == "a"
    assert c.dtype == "int8"
    c = constant(2, dtype="float32")
    assert c.name is None
    assert c.dtype == "float32"


@pytest.mark.parametrize("mode", [Mode("py"), Mode("cvm")])
def test_mean(mode):
    a = iscalar("a")
    b = iscalar("b")
    z = mean(a, b)
    z_fn = aesara.function([a, b], z, mode=mode)
    res = z_fn(1, 1)
    assert np.allclose(res, 1.0)

    a = fscalar("a")
    b = fscalar("b")
    c = fscalar("c")

    z = mean(a, b, c)

    z_fn = aesara.function([a, b, c], aesara.grad(z, [a]), mode=mode)
    res = z_fn(3, 4, 5)
    assert np.allclose(res, 1 / 3)

    z_fn = aesara.function([a, b, c], aesara.grad(z, [b]), mode=mode)
    res = z_fn(3, 4, 5)
    assert np.allclose(res, 1 / 3)

    z = mean()
    z_fn = aesara.function([], z, mode=mode)
    assert z_fn() == 0


def test_shape():
    a = float32("a")
    assert isinstance(a.type, ScalarType)
    assert a.shape.type.ndim == 1
    assert a.shape.type.shape == (0,)
    assert a.shape.type.dtype == "int64"

    b = constant(2, name="b")
    assert isinstance(b.type, ScalarType)
    assert b.shape.type.ndim == 1
    assert b.shape.type.shape == (0,)
    assert b.shape.type.dtype == "int64"


def test_deprecations():
    """Make sure we can import from deprecated modules."""

    with pytest.deprecated_call():
        from aesara.scalar.basic import true_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import true_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import Inv  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import Inv  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import inv  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import inv  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import Scalar  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import Scalar  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import floor_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import floor_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import int_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import int_div  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar.basic import sqr  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.scalar import sqr  # noqa: F401 F811

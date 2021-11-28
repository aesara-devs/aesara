import numpy as np
import pytest

import aesara
from aesara import Mode, function
from aesara.compile.ops import DeepCopyOp
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.misc.safe_asarray import _asarray
from aesara.tensor import get_vector_length
from aesara.tensor.basic import MakeVector, as_tensor_variable, constant
from aesara.tensor.basic_opt import ShapeFeature
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.shape import (
    Reshape,
    Shape_i,
    SpecifyShape,
    reshape,
    shape,
    shape_i,
    specify_shape,
)
from aesara.tensor.subtensor import Subtensor
from aesara.tensor.type import (
    TensorType,
    dmatrix,
    dtensor4,
    dvector,
    fvector,
    ivector,
    matrix,
    scalar,
    tensor3,
    vector,
)
from aesara.tensor.type_other import NoneConst
from aesara.typed_list import make_list
from tests import unittest_tools as utt
from tests.tensor.utils import eval_outputs, random
from tests.test_rop import RopLopChecker


def test_shape_basic():
    s = shape(np.array(1))
    assert np.array_equal(eval_outputs([s]), [])

    s = shape(np.ones((5, 3)))
    assert np.array_equal(eval_outputs([s]), [5, 3])

    s = shape(np.ones(2))
    assert np.array_equal(eval_outputs([s]), [2])

    s = shape(np.ones((5, 3, 10)))
    assert np.array_equal(eval_outputs([s]), [5, 3, 10])


class TestReshape(utt.InferShapeTester, utt.OptimizationTestMixin):
    def setup_method(self):
        self.shared = aesara.shared
        self.op = Reshape
        # The tag canonicalize is needed for the shape test in FAST_COMPILE
        self.mode = None
        self.ignore_topo = (
            DeepCopyOp,
            MakeVector,
            Shape_i,
            DimShuffle,
            Elemwise,
        )
        super().setup_method()

    def function(self, inputs, outputs, ignore_empty=False):
        f = function(inputs, outputs, mode=self.mode)
        if self.mode is not None or config.mode != "FAST_COMPILE":
            topo = f.maker.fgraph.toposort()
            topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
            if ignore_empty:
                assert len(topo_) <= 1, topo_
            else:
                assert len(topo_) == 1, topo_
            if len(topo_) > 0:
                assert type(topo_[0].op) is self.op
        return f

    def test_basics(self):
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
        assert np.array_equal(f_out1, c_val1), (f_out1, c_val1)
        assert np.array_equal(f_out2, c_val2), (f_out2, c_val2)

        # basic to 1 dim(with list)
        c = reshape(b, (as_tensor_variable(6),), ndim=1)
        f = self.function([b], c)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])), np.asarray([0, 1, 2, 3, 4, 5])
        )

        # basic to shape object of same ndim
        c = reshape(b, d.shape)
        f = self.function([b, d], c)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]]), [[0, 1], [2, 3], [4, 5]]),
            np.asarray([[0, 1], [2, 3], [4, 5]]),
        )

        # basic to 2 dims
        c = reshape(a, [2, 3])
        f = self.function([a], c)
        assert np.array_equal(
            f(np.asarray([0, 1, 2, 3, 4, 5])), np.asarray([[0, 1, 2], [3, 4, 5]])
        )

        # test that it works without inplace operations
        a_val = np.asarray([0, 1, 2, 3, 4, 5])
        a_val_copy = np.asarray([0, 1, 2, 3, 4, 5])
        b_val = np.asarray([[0, 1, 2], [3, 4, 5]])

        f_sub = self.function([a, b], c - b)
        assert np.array_equal(f_sub(a_val, b_val), np.zeros_like(b_val))
        assert np.array_equal(a_val, a_val_copy)

        # test that it works with inplace operations
        a_val = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        a_val_copy = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        b_val = _asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")

        f_sub = self.function([a, b], c - b)
        assert np.array_equal(f_sub(a_val, b_val), np.zeros_like(b_val))
        assert np.array_equal(a_val, a_val_copy)

        # verify gradient
        def just_vals(v):
            return Reshape(2)(v, _asarray([2, 3], dtype="int32"))

        utt.verify_grad(just_vals, [a_val], mode=self.mode)

        # test infer_shape
        self._compile_and_check([a], [c], (a_val,), self.op)

        # test broadcast flag for constant value of 1
        c = reshape(b, (b.shape[0], b.shape[1], 1))
        # That reshape may get replaced with a dimshuffle, with is ignored,
        # so we pass "ignore_empty=True"
        f = self.function([b], c, ignore_empty=True)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])),
            np.asarray([[[0], [1], [2]], [[3], [4], [5]]]),
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
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])),
            np.asarray([[[0], [1]], [[2], [3]], [[4], [5]]]),
        )
        assert f.maker.fgraph.toposort()[-1].outputs[0].type.broadcastable == (
            False,
            False,
            True,
        )

    def test_m1(self):
        t = tensor3()
        rng = np.random.default_rng(seed=utt.fetch_seed())
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
        assert np.allclose(r.eval({v: np.arange(5.0)}).T, np.arange(5.0))

    def test_bad_shape(self):
        a = matrix("a")
        shapes = ivector("shapes")
        rng = np.random.default_rng(seed=utt.fetch_seed())
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
        const = constant([1]).reshape(())
        f = function([], const)
        assert f().shape == ()

    def test_more_shapes(self):
        # TODO: generalize infer_shape to account for tensor variable
        # (non-constant) input shape
        admat = dmatrix()
        ndim = 1
        admat_val = random(3, 4)
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
        adtens4_val = random(2, 4, 3, 5)
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


def test_shape_i_hash():
    assert isinstance(Shape_i(np.int64(1)).__hash__(), int)


class TestSpecifyShape(utt.InferShapeTester):
    mode = None
    input_type = TensorType

    def shortDescription(self):
        return None

    def test_check_inputs(self):
        with pytest.raises(AssertionError, match="must be an integer type"):
            specify_shape([[1, 2, 3], [4, 5, 6]], (2.2, 3))
        specify_shape([[1, 2, 3], [4, 5, 6]], (2, 3))

        # Incompatible dimensionality is detected right away
        with pytest.raises(AssertionError, match="will never match"):
            specify_shape(
                matrix(),
                [
                    4,
                ],
            )

    def test_scalar_shapes(self):
        with pytest.raises(AssertionError, match="will never match"):
            specify_shape(vector(), shape=())
        with pytest.raises(AssertionError, match="will never match"):
            specify_shape(matrix(), shape=[])

        x = scalar()
        y = specify_shape(x, shape=())
        f = aesara.function([x], y, mode=self.mode)
        assert f(15) == 15

    def test_python_perform(self):
        x = scalar()
        s = vector(dtype="int32")
        y = specify_shape(x, s)
        f = aesara.function([x, s], y, mode=Mode("py"))
        assert f(12, ()) == 12
        with pytest.raises(
            AssertionError,
            match=r"Got 0 dimensions \(shape \(\)\), expected 1 dimensions with shape \(2,\).",
        ):
            f(12, (2,))

        x = matrix()
        s = vector(dtype="int32")
        y = specify_shape(x, s)
        f = aesara.function([x, s], y, mode=Mode("py"))
        f(np.ones((2, 3)).astype(config.floatX), (2, 3))
        with pytest.raises(
            AssertionError, match=r"Got shape \(3, 4\), expected \(2, 3\)."
        ):
            f(np.ones((3, 4)).astype(config.floatX), (2, 3))

    def test_bad_shape(self):
        # Test that at run time we raise an exception when the shape
        # is not the one specified
        specify_shape = SpecifyShape()

        x = vector()
        xval = np.random.random((2)).astype(config.floatX)
        f = aesara.function([x], specify_shape(x, [2]), mode=self.mode)
        f(xval)
        xval = np.random.random((3)).astype(config.floatX)
        expected = r"(Got shape \(3,\), expected \(2,\))"
        expected += r"|(dim 0 of input has shape 3, expected 2.)"
        with pytest.raises(AssertionError, match=expected):
            f(xval)

        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )

        x = matrix()
        xval = np.random.random((2, 3)).astype(config.floatX)
        f = aesara.function([x], specify_shape(x, [2, 3]), mode=self.mode)
        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )
        f(xval)
        for shape_ in [(4, 3), (2, 8)]:
            xval = np.random.random(shape_).astype(config.floatX)
            s_exp = str(shape_).replace("(", r"\(").replace(")", r"\)")
            expected = rf"(Got shape {s_exp}, expected \(2, 3\).)"
            expected += r"|(dim 0 of input has shape 4, expected 2)"
            expected += r"|(dim 1 of input has shape 8, expected 3)"
            with pytest.raises(AssertionError, match=expected):
                f(xval)

    def test_bad_number_of_shape(self):
        # Test that the number of dimensions provided is good
        specify_shape = SpecifyShape()

        x = vector()
        shape_vec = ivector()
        xval = np.random.random((2)).astype(config.floatX)
        with pytest.raises(AssertionError, match="will never match"):
            specify_shape(x, [])
        with pytest.raises(AssertionError, match="will never match"):
            specify_shape(x, [2, 2])

        f = aesara.function([x, shape_vec], specify_shape(x, shape_vec), mode=self.mode)
        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )
        expected = r"(Got 1 dimensions \(shape \(2,\)\), expected 0 dimensions with shape \(\).)"
        expected += r"|(Got 1 dimensions, expected 0 dimensions.)"
        with pytest.raises(AssertionError, match=expected):
            f(xval, [])
        expected = r"(Got 1 dimensions \(shape \(2,\)\), expected 2 dimensions with shape \(2, 2\).)"
        expected += r"|(SpecifyShape: Got 1 dimensions, expected 2 dimensions.)"
        with pytest.raises(AssertionError, match=expected):
            f(xval, [2, 2])

        x = matrix()
        xval = np.random.random((2, 3)).astype(config.floatX)
        for shape_ in [(), (1,), (2, 3, 4)]:
            with pytest.raises(AssertionError, match="will never match"):
                specify_shape(x, shape_)
            f = aesara.function(
                [x, shape_vec], specify_shape(x, shape_vec), mode=self.mode
            )
            assert isinstance(
                [
                    n
                    for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, SpecifyShape)
                ][0]
                .inputs[0]
                .type,
                self.input_type,
            )
            s_exp = str(shape_).replace("(", r"\(").replace(")", r"\)")
            expected = rf"(Got 2 dimensions \(shape \(2, 3\)\), expected {len(shape_)} dimensions with shape {s_exp}.)"
            expected += rf"|(SpecifyShape: Got 2 dimensions, expected {len(shape_)} dimensions.)"
            with pytest.raises(AssertionError, match=expected):
                f(xval, shape_)

    def test_infer_shape(self):
        rng = np.random.default_rng(3453)
        adtens4 = dtensor4()
        aivec = ivector()
        aivec_val = [3, 4, 2, 5]
        adtens4_val = rng.random(aivec_val)
        self._compile_and_check(
            [adtens4, aivec],
            [SpecifyShape()(adtens4, aivec)],
            [adtens4_val, aivec_val],
            SpecifyShape,
        )


class TestRopLop(RopLopChecker):
    def test_shape(self):
        self.check_nondiff_rop(self.x.shape[0])

    def test_specifyshape(self):
        self.check_rop_lop(specify_shape(self.x, self.in_shape), self.in_shape)

    def test_reshape(self):
        new_shape = constant(
            np.asarray([self.mat_in_shape[0] * self.mat_in_shape[1]], dtype="int64")
        )

        self.check_mat_rop_lop(
            self.mx.reshape(new_shape), (self.mat_in_shape[0] * self.mat_in_shape[1],)
        )


@config.change_flags(compute_test_value="raise")
def test_nonstandard_shapes():
    a = tensor3(config.floatX)
    a.tag.test_value = np.random.random((2, 3, 4)).astype(config.floatX)
    b = tensor3(config.floatX)
    b.tag.test_value = np.random.random((2, 3, 4)).astype(config.floatX)

    tl = make_list([a, b])
    tl_shape = shape(tl)
    assert np.array_equal(tl_shape.get_test_value(), (2, 2, 3, 4))

    # There's no `FunctionGraph`, so it should return a `Subtensor`
    tl_shape_i = shape_i(tl, 0)
    assert isinstance(tl_shape_i.owner.op, Subtensor)
    assert tl_shape_i.get_test_value() == 2

    tl_fg = FunctionGraph([a, b], [tl], features=[ShapeFeature()])
    tl_shape_i = shape_i(tl, 0, fgraph=tl_fg)
    assert not isinstance(tl_shape_i.owner.op, Subtensor)
    assert tl_shape_i.get_test_value() == 2

    none_shape = shape(NoneConst)
    assert np.array_equal(none_shape.get_test_value(), [])


def test_shape_i_basics():
    with pytest.raises(TypeError):
        Shape_i(0)([1, 2])

    with pytest.raises(TypeError):
        Shape_i(0)(scalar())


def test_get_vector_length():
    # Test `Shape`s
    x = aesara.shared(np.zeros((2, 3, 4, 5)))
    assert get_vector_length(x.shape) == 4

    # Test `SpecifyShape`
    x = specify_shape(ivector(), (10,))
    assert get_vector_length(x) == 10

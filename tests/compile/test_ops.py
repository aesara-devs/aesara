import pickle

import numpy as np
import pytest

import theano
from tests import unittest_tools as utt
from theano import change_flags, config, function
from theano.compile.ops import Rebroadcast, SpecifyShape, as_op, shape, shape_i
from theano.gof.fg import FunctionGraph
from theano.tensor.basic import (
    TensorType,
    dmatrix,
    dtensor4,
    dvector,
    ivector,
    matrix,
    tensor3,
    vector,
)
from theano.tensor.opt import ShapeFeature
from theano.tensor.subtensor import Subtensor
from theano.tensor.type_other import NoneConst
from theano.typed_list import make_list


@as_op([dmatrix, dmatrix], dmatrix)
def mul(a, b):
    """
    This is for test_pickle, since the function still has to be
    reachable from pickle (as in it cannot be defined inline)
    """
    return a * b


class TestOpDecorator(utt.InferShapeTester):
    def test_1arg(self):
        x = dmatrix("x")

        @as_op(dmatrix, dvector)
        def cumprod(x):
            return np.cumprod(x)

        fn = function([x], cumprod(x))
        r = fn([[1.5, 5], [2, 2]])
        r0 = np.array([1.5, 7.5, 15.0, 30.0])

        assert np.allclose(r, r0), (r, r0)

    def test_2arg(self):
        x = dmatrix("x")
        x.tag.test_value = np.zeros((2, 2))
        y = dvector("y")
        y.tag.test_value = [0, 0, 0, 0]

        @as_op([dmatrix, dvector], dvector)
        def cumprod_plus(x, y):
            return np.cumprod(x) + y

        fn = function([x, y], cumprod_plus(x, y))
        r = fn([[1.5, 5], [2, 2]], [1, 100, 2, 200])
        r0 = np.array([2.5, 107.5, 17.0, 230.0])

        assert np.allclose(r, r0), (r, r0)

    def test_infer_shape(self):
        x = dmatrix("x")
        x.tag.test_value = np.zeros((2, 2))
        y = dvector("y")
        y.tag.test_value = [0, 0, 0, 0]

        def infer_shape(node, shapes):
            x, y = shapes
            return [y]

        @as_op([dmatrix, dvector], dvector, infer_shape)
        def cumprod_plus(x, y):
            return np.cumprod(x) + y

        self._compile_and_check(
            [x, y],
            [cumprod_plus(x, y)],
            [[[1.5, 5], [2, 2]], [1, 100, 2, 200]],
            cumprod_plus.__class__,
            warn=False,
        )

    def test_pickle(self):
        x = dmatrix("x")
        y = dmatrix("y")

        m = mul(x, y)

        s = pickle.dumps(m)
        m2 = pickle.loads(s)

        assert m2.owner.op == m.owner.op


def test_shape_i_hash():
    assert isinstance(theano.tensor.opt.Shape_i(np.int64(1)).__hash__(), int)


class TestSpecifyShape(utt.InferShapeTester):
    mode = None
    input_type = TensorType

    def shortDescription(self):
        return None

    def test_bad_shape(self):
        # Test that at run time we raise an exception when the shape
        # is not the one specified
        specify_shape = SpecifyShape()

        x = vector()
        xval = np.random.rand(2).astype(config.floatX)
        f = theano.function([x], specify_shape(x, [2]), mode=self.mode)
        f(xval)
        xval = np.random.rand(3).astype(config.floatX)
        with pytest.raises(AssertionError):
            f(xval)

        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )

        x = matrix()
        xval = np.random.rand(2, 3).astype(config.floatX)
        f = theano.function([x], specify_shape(x, [2, 3]), mode=self.mode)
        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )
        f(xval)
        for shape_ in [(1, 3), (2, 2), (5, 5)]:
            xval = np.random.rand(*shape_).astype(config.floatX)
            with pytest.raises(AssertionError):
                f(xval)

    def test_bad_number_of_shape(self):
        # Test that the number of dimensions provided is good
        specify_shape = SpecifyShape()

        x = vector()
        shape_vec = ivector()
        xval = np.random.rand(2).astype(config.floatX)
        with pytest.raises(AssertionError):
            specify_shape(x, [])
        with pytest.raises(AssertionError):
            specify_shape(x, [2, 2])

        f = theano.function([x, shape_vec], specify_shape(x, shape_vec), mode=self.mode)
        assert isinstance(
            [n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape)][0]
            .inputs[0]
            .type,
            self.input_type,
        )
        with pytest.raises(AssertionError):
            f(xval, [])
        with pytest.raises(AssertionError):
            f(xval, [2, 2])

        x = matrix()
        xval = np.random.rand(2, 3).astype(config.floatX)
        for shape_ in [(), (1,), (2, 3, 4)]:
            with pytest.raises(AssertionError):
                specify_shape(x, shape_)
            f = theano.function(
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
            with pytest.raises(AssertionError):
                f(xval, shape_)

    def test_infer_shape(self):
        rng = np.random.RandomState(3453)
        adtens4 = dtensor4()
        aivec = ivector()
        aivec_val = [3, 4, 2, 5]
        adtens4_val = rng.rand(*aivec_val)
        self._compile_and_check(
            [adtens4, aivec],
            [SpecifyShape()(adtens4, aivec)],
            [adtens4_val, aivec_val],
            SpecifyShape,
        )


class TestRebroadcast(utt.InferShapeTester):
    def test_rebroadcast(self):
        rng = np.random.RandomState(3453)
        # Rebroadcast
        adtens4 = dtensor4()
        adict = [(0, False), (1, True), (2, False), (3, True)]
        adtens4_val = rng.rand(2, 1, 3, 1).astype(config.floatX)
        self._compile_and_check(
            [adtens4],
            [Rebroadcast(*adict)(adtens4)],
            [adtens4_val],
            Rebroadcast,
            warn=False,
        )

        adtens4_bro = TensorType("float64", (True, True, True, False))()
        bdict = [(0, True), (1, False), (2, False), (3, False)]
        adtens4_bro_val = rng.rand(1, 1, 1, 3).astype(config.floatX)
        self._compile_and_check(
            [adtens4_bro],
            [Rebroadcast(*bdict)(adtens4_bro)],
            [adtens4_bro_val],
            Rebroadcast,
        )


@change_flags(compute_test_value="raise")
def test_nonstandard_shapes():
    a = tensor3(config.floatX)
    a.tag.test_value = np.random.random((2, 3, 4)).astype(config.floatX)
    b = tensor3(theano.config.floatX)
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

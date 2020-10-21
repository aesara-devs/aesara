import pytest


sp = pytest.importorskip("scipy", minversion="0.7.0")

import numpy as np

import theano
from tests import unittest_tools as utt
from tests.sparse.test_basic import as_sparse_format
from theano import config, sparse, tensor
from theano.sparse.sandbox.sp2 import (
    Binomial,
    Multinomial,
    Poisson,
    multinomial,
    poisson,
)


class TestPoisson(utt.InferShapeTester):
    x = {}
    a = {}

    for format in sparse.sparse_formats:
        variable = getattr(theano.sparse, format + "_matrix")

        rand = np.array(
            np.random.randint(1, 4, size=(3, 4)) - 1, dtype=theano.config.floatX
        )

        x[format] = variable()
        a[format] = as_sparse_format(rand, format)

    def setup_method(self):
        super().setup_method()
        self.op_class = Poisson

    def test_op(self):
        for format in sparse.sparse_formats:
            f = theano.function([self.x[format]], poisson(self.x[format]))

            tested = f(self.a[format])

            assert tested.format == format
            assert tested.dtype == self.a[format].dtype
            assert np.allclose(np.floor(tested.data), tested.data)
            assert tested.shape == self.a[format].shape

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(
                [self.x[format]],
                [poisson(self.x[format])],
                [self.a[format]],
                self.op_class,
            )


class TestBinomial(utt.InferShapeTester):
    n = tensor.scalar(dtype="int64")
    p = tensor.scalar()
    shape = tensor.lvector()
    _n = 5
    _p = 0.25
    _shape = np.asarray([3, 5], dtype="int64")

    inputs = [n, p, shape]
    _inputs = [_n, _p, _shape]

    def setup_method(self):
        super().setup_method()
        self.op_class = Binomial

    def test_op(self):
        for sp_format in sparse.sparse_formats:
            for o_type in sparse.float_dtypes:
                f = theano.function(
                    self.inputs, Binomial(sp_format, o_type)(*self.inputs)
                )

                tested = f(*self._inputs)

                assert tested.shape == tuple(self._shape)
                assert tested.format == sp_format
                assert tested.dtype == o_type
                assert np.allclose(np.floor(tested.todense()), tested.todense())

    def test_infer_shape(self):
        for sp_format in sparse.sparse_formats:
            for o_type in sparse.float_dtypes:
                self._compile_and_check(
                    self.inputs,
                    [Binomial(sp_format, o_type)(*self.inputs)],
                    self._inputs,
                    self.op_class,
                )


class TestMultinomial(utt.InferShapeTester):
    p = sparse.csr_matrix()
    _p = sp.sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.5, 0.0, 0.5],
                [0.1, 0.2, 0.3, 0.4],
                [0.0, 1.0, 0.0, 0.0],
                [0.3, 0.3, 0.0, 0.4],
            ],
            dtype=config.floatX,
        )
    )

    def setup_method(self):
        super().setup_method()
        self.op_class = Multinomial

    def test_op(self):
        n = tensor.lscalar()
        f = theano.function([self.p, n], multinomial(n, self.p))

        _n = 5
        tested = f(self._p, _n)
        assert tested.shape == self._p.shape
        assert np.allclose(np.floor(tested.todense()), tested.todense())
        assert tested[2, 1] == _n

        n = tensor.lvector()
        f = theano.function([self.p, n], multinomial(n, self.p))

        _n = np.asarray([1, 2, 3, 4], dtype="int64")
        tested = f(self._p, _n)
        assert tested.shape == self._p.shape
        assert np.allclose(np.floor(tested.todense()), tested.todense())
        assert tested[2, 1] == _n[2]

    def test_infer_shape(self):
        self._compile_and_check(
            [self.p], [multinomial(5, self.p)], [self._p], self.op_class, warn=False
        )

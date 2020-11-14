import numpy as np
import pytest

import theano
from tests import unittest_tools as utt
from theano import change_flags, config, function
from theano import tensor as tt
from theano.tensor.extra_ops import (
    Bartlett,
    BroadcastTo,
    CpuContiguous,
    CumOp,
    DiffOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    RepeatOp,
    SearchsortedOp,
    Unique,
    UnravelIndex,
    bartlett,
    bincount,
    broadcast_shape,
    broadcast_to,
    compress,
    cpu_contiguous,
    cumprod,
    cumsum,
    diff,
    fill_diagonal,
    fill_diagonal_offset,
    ravel_multi_index,
    repeat,
    searchsorted,
    squeeze,
    to_one_hot,
    unravel_index,
)


def test_cpu_contiguous():
    a = tt.fmatrix("a")
    i = tt.iscalar("i")
    a_val = np.asarray(np.random.rand(4, 5), dtype="float32")
    f = theano.function([a, i], cpu_contiguous(a.reshape((5, 4))[::i]))
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(node.op, CpuContiguous) for node in topo])
    assert f(a_val, 1).flags["C_CONTIGUOUS"]
    assert f(a_val, 2).flags["C_CONTIGUOUS"]
    assert f(a_val, 3).flags["C_CONTIGUOUS"]
    # Test the grad:

    utt.verify_grad(cpu_contiguous, [np.random.rand(5, 7, 2)])


class TestSearchsortedOp(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = SearchsortedOp
        self.op = SearchsortedOp()

        self.x = tt.vector("x")
        self.v = tt.tensor3("v")

        self.a = 30 * np.random.random(50).astype(config.floatX)
        self.b = 30 * np.random.random((8, 10, 5)).astype(config.floatX)
        self.idx_sorted = np.argsort(self.a).astype("int32")

    def test_searchsortedOp_on_sorted_input(self):
        f = theano.function([self.x, self.v], searchsorted(self.x, self.v))
        assert np.allclose(
            np.searchsorted(self.a[self.idx_sorted], self.b),
            f(self.a[self.idx_sorted], self.b),
        )

        sorter = tt.vector("sorter", dtype="int32")
        f = theano.function(
            [self.x, self.v, sorter],
            self.x.searchsorted(self.v, sorter=sorter, side="right"),
        )
        assert np.allclose(
            self.a.searchsorted(self.b, sorter=self.idx_sorted, side="right"),
            f(self.a, self.b, self.idx_sorted),
        )

        sa = self.a[self.idx_sorted]
        f = theano.function([self.x, self.v], self.x.searchsorted(self.v, side="right"))
        assert np.allclose(sa.searchsorted(self.b, side="right"), f(sa, self.b))

    def test_searchsortedOp_wrong_side_kwd(self):
        with pytest.raises(ValueError):
            searchsorted(self.x, self.v, side="asdfa")

    def test_searchsortedOp_on_no_1d_inp(self):
        no_1d = tt.dmatrix("no_1d")
        with pytest.raises(ValueError):
            searchsorted(no_1d, self.v)
        with pytest.raises(ValueError):
            searchsorted(self.x, self.v, sorter=no_1d)

    def test_searchsortedOp_on_float_sorter(self):
        sorter = tt.vector("sorter", dtype="float32")
        with pytest.raises(TypeError):
            searchsorted(self.x, self.v, sorter=sorter)

    def test_searchsortedOp_on_int_sorter(self):
        compatible_types = ("int8", "int16", "int32")
        if theano.configdefaults.python_int_bitwidth() == 64:
            compatible_types += ("int64",)
        # 'uint8', 'uint16', 'uint32', 'uint64')
        for dtype in compatible_types:
            sorter = tt.vector("sorter", dtype=dtype)
            f = theano.function(
                [self.x, self.v, sorter],
                searchsorted(self.x, self.v, sorter=sorter),
                allow_input_downcast=True,
            )
            assert np.allclose(
                np.searchsorted(self.a, self.b, sorter=self.idx_sorted),
                f(self.a, self.b, self.idx_sorted),
            )

    def test_searchsortedOp_on_right_side(self):
        f = theano.function(
            [self.x, self.v], searchsorted(self.x, self.v, side="right")
        )
        assert np.allclose(
            np.searchsorted(self.a, self.b, side="right"), f(self.a, self.b)
        )

    def test_infer_shape(self):
        # Test using default parameters' value
        self._compile_and_check(
            [self.x, self.v],
            [searchsorted(self.x, self.v)],
            [self.a[self.idx_sorted], self.b],
            self.op_class,
        )

        # Test parameter ``sorter``
        sorter = tt.vector("sorter", dtype="int32")
        self._compile_and_check(
            [self.x, self.v, sorter],
            [searchsorted(self.x, self.v, sorter=sorter)],
            [self.a, self.b, self.idx_sorted],
            self.op_class,
        )

        # Test parameter ``side``
        la = np.ones(10).astype(config.floatX)
        lb = np.ones(shape=(1, 2, 3)).astype(config.floatX)
        self._compile_and_check(
            [self.x, self.v],
            [searchsorted(self.x, self.v, side="right")],
            [la, lb],
            self.op_class,
        )

    def test_grad(self):
        utt.verify_grad(self.op, [self.a[self.idx_sorted], self.b])


class TestCumOp(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = CumOp
        self.op = CumOp()

    def test_cum_op(self):
        x = tt.tensor3("x")
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        # Test axis out of bounds
        with pytest.raises(ValueError):
            cumsum(x, axis=3)
        with pytest.raises(ValueError):
            cumsum(x, axis=-4)
        with pytest.raises(ValueError):
            cumprod(x, axis=3)
        with pytest.raises(ValueError):
            cumprod(x, axis=-4)

        f = theano.function([x], [cumsum(x), cumprod(x)])
        s, p = f(a)
        assert np.allclose(np.cumsum(a), s)  # Test axis=None
        assert np.allclose(np.cumprod(a), p)  # Test axis=None

        for axis in range(-len(a.shape), len(a.shape)):
            f = theano.function([x], [cumsum(x, axis=axis), cumprod(x, axis=axis)])
            s, p = f(a)
            assert np.allclose(np.cumsum(a, axis=axis), s)
            assert np.allclose(np.cumprod(a, axis=axis), p)

    def test_infer_shape(self):
        x = tt.tensor3("x")
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        # Test axis=None
        self._compile_and_check([x], [self.op(x)], [a], self.op_class)

        for axis in range(-len(a.shape), len(a.shape)):
            self._compile_and_check([x], [cumsum(x, axis=axis)], [a], self.op_class)

    def test_grad(self):
        a = np.random.random((3, 5, 2)).astype(config.floatX)

        utt.verify_grad(self.op_class(mode="add"), [a])  # Test axis=None
        utt.verify_grad(self.op_class(mode="mul"), [a])  # Test axis=None

        for axis in range(-len(a.shape), len(a.shape)):
            utt.verify_grad(self.op_class(axis=axis, mode="add"), [a], eps=4e-4)
            utt.verify_grad(self.op_class(axis=axis, mode="mul"), [a], eps=4e-4)


class TestBinCount(utt.InferShapeTester):
    def test_bincountFn(self):
        w = tt.vector("w")

        def ref(data, w=None, minlength=None):
            size = int(data.max() + 1)
            if minlength:
                size = max(size, minlength)
            if w is not None:
                out = np.zeros(size, dtype=w.dtype)
                for i in range(data.shape[0]):
                    out[data[i]] += w[i]
            else:
                out = np.zeros(size, dtype=a.dtype)
                for i in range(data.shape[0]):
                    out[data[i]] += 1
            return out

        for dtype in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            x = tt.vector("x", dtype=dtype)

            a = np.random.randint(1, 51, size=(25)).astype(dtype)
            weights = np.random.random((25,)).astype(config.floatX)

            f1 = theano.function([x], bincount(x))
            f2 = theano.function([x, w], bincount(x, weights=w))

            assert (ref(a) == f1(a)).all()
            assert np.allclose(ref(a, weights), f2(a, weights))
            f3 = theano.function([x], bincount(x, minlength=55))
            f4 = theano.function([x], bincount(x, minlength=5))
            assert (ref(a, minlength=55) == f3(a)).all()
            assert (ref(a, minlength=5) == f4(a)).all()
            # skip the following test when using unsigned ints
            if not dtype.startswith("u"):
                a[0] = -1
                f5 = theano.function([x], bincount(x, assert_nonneg=True))
                with pytest.raises(AssertionError):
                    f5(a)


class TestDiffOp(utt.InferShapeTester):
    nb = 10  # Number of time iterating for n

    def setup_method(self):
        super().setup_method()
        self.op_class = DiffOp
        self.op = DiffOp()

    def test_diffOp(self):
        x = tt.matrix("x")
        a = np.random.random((30, 50)).astype(config.floatX)

        f = theano.function([x], diff(x))
        assert np.allclose(np.diff(a), f(a))

        for axis in range(len(a.shape)):
            for k in range(TestDiffOp.nb):
                g = theano.function([x], diff(x, n=k, axis=axis))
                assert np.allclose(np.diff(a, n=k, axis=axis), g(a))

    def test_infer_shape(self):
        x = tt.matrix("x")
        a = np.random.random((30, 50)).astype(config.floatX)

        self._compile_and_check([x], [self.op(x)], [a], self.op_class)

        for axis in range(len(a.shape)):
            for k in range(TestDiffOp.nb):
                self._compile_and_check(
                    [x], [diff(x, n=k, axis=axis)], [a], self.op_class
                )

    def test_grad(self):
        x = tt.vector("x")
        a = np.random.random(50).astype(config.floatX)

        theano.function([x], tt.grad(tt.sum(diff(x)), x))
        utt.verify_grad(self.op, [a])

        for k in range(TestDiffOp.nb):
            theano.function([x], tt.grad(tt.sum(diff(x, n=k)), x))
            utt.verify_grad(DiffOp(n=k), [a], eps=7e-3)


class TestSqueeze(utt.InferShapeTester):
    shape_list = [(1, 3), (1, 2, 3), (1, 5, 1, 1, 6)]
    broadcast_list = [
        [True, False],
        [True, False, False],
        [True, False, True, True, False],
    ]

    def setup_method(self):
        super().setup_method()
        self.op = squeeze

    def test_op(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tt.TensorType(theano.config.floatX, broadcast)()

            f = theano.function([variable], self.op(variable))

            expected = np.squeeze(data)
            tested = f(data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)

    def test_infer_shape(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tt.TensorType(theano.config.floatX, broadcast)()

            self._compile_and_check(
                [variable], [self.op(variable)], [data], tt.DimShuffle, warn=False
            )

    def test_grad(self):
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)

            utt.verify_grad(self.op, [data])

    def test_var_interface(self):
        # same as test_op, but use a_theano_var.squeeze.
        for shape, broadcast in zip(self.shape_list, self.broadcast_list):
            data = np.random.random(size=shape).astype(theano.config.floatX)
            variable = tt.TensorType(theano.config.floatX, broadcast)()

            f = theano.function([variable], variable.squeeze())

            expected = np.squeeze(data)
            tested = f(data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)

    def test_axis(self):
        variable = tt.TensorType(theano.config.floatX, [False, True, False])()
        res = squeeze(variable, axis=1)

        assert res.broadcastable == (False, False)

        variable = tt.TensorType(theano.config.floatX, [False, True, False])()
        res = squeeze(variable, axis=(1,))

        assert res.broadcastable == (False, False)

        variable = tt.TensorType(theano.config.floatX, [False, True, False, True])()
        res = squeeze(variable, axis=(1, 3))

        assert res.broadcastable == (False, False)


class TestCompress(utt.InferShapeTester):
    axis_list = [None, -1, 0, 0, 0, 1]
    cond_list = [
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [],
        [0, 0, 0, 0],
        [1, 1, 0, 1, 0],
    ]
    shape_list = [(2, 3), (4, 3), (4, 3), (4, 3), (4, 3), (3, 5)]

    def setup_method(self):
        super().setup_method()
        self.op = compress

    def test_op(self):
        for axis, cond, shape in zip(self.axis_list, self.cond_list, self.shape_list):
            cond_var = theano.tensor.ivector()
            data = np.random.random(size=shape).astype(theano.config.floatX)
            data_var = theano.tensor.matrix()

            f = theano.function(
                [cond_var, data_var], self.op(cond_var, data_var, axis=axis)
            )

            expected = np.compress(cond, data, axis=axis)
            tested = f(cond, data)

            assert tested.shape == expected.shape
            assert np.allclose(tested, expected)


class TestRepeatOp(utt.InferShapeTester):
    def _possible_axis(self, ndim):
        return [None] + list(range(ndim)) + [-i for i in range(ndim)]

    def setup_method(self):
        super().setup_method()
        self.op_class = RepeatOp
        self.op = RepeatOp()
        # uint64 always fails
        # int64 and uint32 also fail if python int are 32-bit
        ptr_bitwidth = theano.configdefaults.local_bitwidth()
        if ptr_bitwidth == 64:
            self.numpy_unsupported_dtypes = ("uint64",)
        if ptr_bitwidth == 32:
            self.numpy_unsupported_dtypes = ("uint32", "int64", "uint64")

    def test_repeatOp(self):
        for ndim in [1, 3]:
            x = tt.TensorType(config.floatX, [False] * ndim)()
            a = np.random.random((10,) * ndim).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                for dtype in tt.integer_dtypes:
                    r_var = tt.scalar(dtype=dtype)
                    r = np.asarray(3, dtype=dtype)
                    if dtype == "uint64" or (
                        dtype in self.numpy_unsupported_dtypes and r_var.ndim == 1
                    ):
                        with pytest.raises(TypeError):
                            repeat(x, r_var, axis=axis)
                    else:
                        f = theano.function([x, r_var], repeat(x, r_var, axis=axis))
                        assert np.allclose(np.repeat(a, r, axis=axis), f(a, r))

                        r_var = tt.vector(dtype=dtype)
                        if axis is None:
                            r = np.random.randint(1, 6, size=a.size).astype(dtype)
                        else:
                            r = np.random.randint(1, 6, size=(10,)).astype(dtype)

                        if dtype in self.numpy_unsupported_dtypes and r_var.ndim == 1:
                            with pytest.raises(TypeError):
                                repeat(x, r_var, axis=axis)
                        else:
                            f = theano.function([x, r_var], repeat(x, r_var, axis=axis))
                            assert np.allclose(np.repeat(a, r, axis=axis), f(a, r))

                        # check when r is a list of single integer, e.g. [3].
                        r = np.random.randint(1, 11, size=()).astype(dtype) + 2
                        f = theano.function([x], repeat(x, [r], axis=axis))
                        assert np.allclose(np.repeat(a, r, axis=axis), f(a))
                        assert not np.any(
                            [
                                isinstance(n.op, RepeatOp)
                                for n in f.maker.fgraph.toposort()
                            ]
                        )

                        # check when r is  theano tensortype that broadcastable is (True,)
                        r_var = theano.tensor.TensorType(
                            broadcastable=(True,), dtype=dtype
                        )()
                        r = np.random.randint(1, 6, size=(1,)).astype(dtype)
                        f = theano.function([x, r_var], repeat(x, r_var, axis=axis))
                        assert np.allclose(np.repeat(a, r[0], axis=axis), f(a, r))
                        assert not np.any(
                            [
                                isinstance(n.op, RepeatOp)
                                for n in f.maker.fgraph.toposort()
                            ]
                        )

    @pytest.mark.slow
    def test_infer_shape(self):
        for ndim in [1, 3]:
            x = tt.TensorType(config.floatX, [False] * ndim)()
            shp = (np.arange(ndim) + 1) * 3
            a = np.random.random(shp).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                for dtype in ["int8", "uint8", "uint64"]:
                    r_var = tt.scalar(dtype=dtype)
                    r = np.asarray(3, dtype=dtype)
                    if dtype in self.numpy_unsupported_dtypes:
                        r_var = tt.vector(dtype=dtype)
                        with pytest.raises(TypeError):
                            repeat(x, r_var)
                    else:
                        self._compile_and_check(
                            [x, r_var],
                            [RepeatOp(axis=axis)(x, r_var)],
                            [a, r],
                            self.op_class,
                        )

                        r_var = tt.vector(dtype=dtype)
                        if axis is None:
                            r = np.random.randint(1, 6, size=a.size).astype(dtype)
                        elif a.size > 0:
                            r = np.random.randint(1, 6, size=a.shape[axis]).astype(
                                dtype
                            )
                        else:
                            r = np.random.randint(1, 6, size=(10,)).astype(dtype)

                        self._compile_and_check(
                            [x, r_var],
                            [RepeatOp(axis=axis)(x, r_var)],
                            [a, r],
                            self.op_class,
                        )

    def test_grad(self):
        for ndim in range(3):
            a = np.random.random((10,) * ndim).astype(config.floatX)

            for axis in self._possible_axis(ndim):
                utt.verify_grad(lambda x: RepeatOp(axis=axis)(x, 3), [a])

    def test_broadcastable(self):
        x = tt.TensorType(config.floatX, [False, True, False])()
        r = RepeatOp(axis=1)(x, 2)
        assert r.broadcastable == (False, False, False)
        r = RepeatOp(axis=1)(x, 1)
        assert r.broadcastable == (False, True, False)
        r = RepeatOp(axis=0)(x, 2)
        assert r.broadcastable == (False, True, False)


class TestBartlett(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Bartlett
        self.op = bartlett

    def test_perform(self):
        x = tt.lscalar()
        f = function([x], self.op(x))
        M = np.random.randint(3, 51, size=())
        assert np.allclose(f(M), np.bartlett(M))
        assert np.allclose(f(0), np.bartlett(0))
        assert np.allclose(f(-1), np.bartlett(-1))
        b = np.array([17], dtype="uint8")
        assert np.allclose(f(b[0]), np.bartlett(b[0]))

    def test_infer_shape(self):
        x = tt.lscalar()
        self._compile_and_check(
            [x], [self.op(x)], [np.random.randint(3, 51, size=())], self.op_class
        )
        self._compile_and_check([x], [self.op(x)], [0], self.op_class)
        self._compile_and_check([x], [self.op(x)], [1], self.op_class)


class TestFillDiagonal(utt.InferShapeTester):

    rng = np.random.RandomState(43)

    def setup_method(self):
        super().setup_method()
        self.op_class = FillDiagonal
        self.op = fill_diagonal

    def test_perform(self):
        x = tt.matrix()
        y = tt.scalar()
        f = function([x, y], fill_diagonal(x, y))
        for shp in [(8, 8), (5, 8), (8, 5)]:
            a = np.random.rand(*shp).astype(config.floatX)
            val = np.cast[config.floatX](np.random.rand())
            out = f(a, val)
            # We can't use np.fill_diagonal as it is bugged.
            assert np.allclose(np.diag(out), val)
            assert (out == val).sum() == min(a.shape)

        # test for 3dtt
        a = np.random.rand(3, 3, 3).astype(config.floatX)
        x = tt.tensor3()
        y = tt.scalar()
        f = function([x, y], fill_diagonal(x, y))
        val = np.cast[config.floatX](np.random.rand() + 10)
        out = f(a, val)
        # We can't use np.fill_diagonal as it is bugged.
        assert out[0, 0, 0] == val
        assert out[1, 1, 1] == val
        assert out[2, 2, 2] == val
        assert (out == val).sum() == min(a.shape)

    @pytest.mark.slow
    def test_gradient(self):
        utt.verify_grad(
            fill_diagonal,
            [np.random.rand(5, 8), np.random.rand()],
            n_tests=1,
            rng=TestFillDiagonal.rng,
        )
        utt.verify_grad(
            fill_diagonal,
            [np.random.rand(8, 5), np.random.rand()],
            n_tests=1,
            rng=TestFillDiagonal.rng,
        )

    def test_infer_shape(self):
        z = tt.dtensor3()
        x = tt.dmatrix()
        y = tt.dscalar()
        self._compile_and_check(
            [x, y],
            [self.op(x, y)],
            [np.random.rand(8, 5), np.random.rand()],
            self.op_class,
        )
        self._compile_and_check(
            [z, y],
            [self.op(z, y)],
            # must be square when nd>2
            [np.random.rand(8, 8, 8), np.random.rand()],
            self.op_class,
            warn=False,
        )


class TestFillDiagonalOffset(utt.InferShapeTester):

    rng = np.random.RandomState(43)

    def setup_method(self):
        super().setup_method()
        self.op_class = FillDiagonalOffset
        self.op = fill_diagonal_offset

    def test_perform(self):
        x = tt.matrix()
        y = tt.scalar()
        z = tt.iscalar()

        f = function([x, y, z], fill_diagonal_offset(x, y, z))
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            for shp in [(8, 8), (5, 8), (8, 5), (5, 5)]:
                a = np.random.rand(*shp).astype(config.floatX)
                val = np.cast[config.floatX](np.random.rand())
                out = f(a, val, test_offset)
                # We can't use np.fill_diagonal as it is bugged.
                assert np.allclose(np.diag(out, test_offset), val)
                if test_offset >= 0:
                    assert (out == val).sum() == min(
                        min(a.shape), a.shape[1] - test_offset
                    )
                else:
                    assert (out == val).sum() == min(
                        min(a.shape), a.shape[0] + test_offset
                    )

    def test_gradient(self):
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            # input 'offset' will not be tested
            def fill_diagonal_with_fix_offset(a, val):
                return fill_diagonal_offset(a, val, test_offset)

            utt.verify_grad(
                fill_diagonal_with_fix_offset,
                [np.random.rand(5, 8), np.random.rand()],
                n_tests=1,
                rng=TestFillDiagonalOffset.rng,
            )
            utt.verify_grad(
                fill_diagonal_with_fix_offset,
                [np.random.rand(8, 5), np.random.rand()],
                n_tests=1,
                rng=TestFillDiagonalOffset.rng,
            )
            utt.verify_grad(
                fill_diagonal_with_fix_offset,
                [np.random.rand(5, 5), np.random.rand()],
                n_tests=1,
                rng=TestFillDiagonalOffset.rng,
            )

    def test_infer_shape(self):
        x = tt.dmatrix()
        y = tt.dscalar()
        z = tt.iscalar()
        for test_offset in (-5, -4, -1, 0, 1, 4, 5):
            self._compile_and_check(
                [x, y, z],
                [self.op(x, y, z)],
                [np.random.rand(8, 5), np.random.rand(), test_offset],
                self.op_class,
            )
            self._compile_and_check(
                [x, y, z],
                [self.op(x, y, z)],
                [np.random.rand(5, 8), np.random.rand(), test_offset],
                self.op_class,
            )


def test_to_one_hot():
    v = theano.tensor.ivector()
    o = to_one_hot(v, 10)
    f = theano.function([v], o)
    out = f([1, 2, 3, 5, 6])
    assert out.dtype == theano.config.floatX
    assert np.allclose(
        out,
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
    )

    v = theano.tensor.ivector()
    o = to_one_hot(v, 10, dtype="int32")
    f = theano.function([v], o)
    out = f([1, 2, 3, 5, 6])
    assert out.dtype == "int32"
    assert np.allclose(
        out,
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
    )


class TestUnique(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Unique
        self.ops = [
            Unique(),
            Unique(True),
            Unique(False, True),
            Unique(True, True),
            Unique(False, False, True),
            Unique(True, False, True),
            Unique(False, True, True),
            Unique(True, True, True),
        ]

    def test_basic_vector(self):
        # Basic test for a vector.
        # Done by using the op and checking that it returns the right answer.

        x = theano.tensor.vector()
        inp = np.asarray([2, 1, 3, 2], dtype=config.floatX)
        list_outs_expected = [
            [np.unique(inp)],
            np.unique(inp, True),
            np.unique(inp, False, True),
            np.unique(inp, True, True),
            np.unique(inp, False, False, True),
            np.unique(inp, True, False, True),
            np.unique(inp, False, True, True),
            np.unique(inp, True, True, True),
        ]
        for op, outs_expected in zip(self.ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_basic_matrix(self):
        # Basic test for a matrix.
        # Done by using the op and checking that it returns the right answer.

        x = theano.tensor.matrix()
        inp = np.asarray([[2, 1], [3, 2], [2, 1]], dtype=config.floatX)
        list_outs_expected = [
            [np.unique(inp)],
            np.unique(inp, True),
            np.unique(inp, False, True),
            np.unique(inp, True, True),
            np.unique(inp, False, False, True),
            np.unique(inp, True, False, True),
            np.unique(inp, False, True, True),
            np.unique(inp, True, True, True),
        ]
        for op, outs_expected in zip(self.ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_infer_shape_vector(self):
        # Testing the infer_shape with a vector.

        x = theano.tensor.vector()

        for op in self.ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check(
                [x],
                [f],
                [np.asarray(np.array([2, 1, 3, 2]), dtype=config.floatX)],
                self.op_class,
            )

    def test_infer_shape_matrix(self):
        # Testing the infer_shape with a matrix.

        x = theano.tensor.matrix()

        for op in self.ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check(
                [x],
                [f],
                [np.asarray(np.array([[2, 1], [3, 2], [2, 3]]), dtype=config.floatX)],
                self.op_class,
            )


class TestUniqueAxis(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        numpy_ver = tuple([int(n) for n in np.__version__.split(".")])
        if numpy_ver >= (1, 13):
            self.expect_success = True
        else:
            self.expect_success = False
        self.ops_pars = [
            (tuple(), {"axis": 0}),
            ((True,), {"axis": 0}),
            (
                (
                    False,
                    True,
                ),
                {"axis": 0},
            ),
            (
                (
                    True,
                    True,
                ),
                {"axis": 0},
            ),
            (
                (
                    False,
                    False,
                    True,
                ),
                {"axis": 0},
            ),
            (
                (
                    True,
                    False,
                    True,
                ),
                {"axis": 0},
            ),
            (
                (
                    False,
                    True,
                    True,
                ),
                {"axis": 0},
            ),
            (
                (
                    True,
                    True,
                    True,
                ),
                {"axis": 0},
            ),
            (tuple(), {"axis": -1}),
            ((True,), {"axis": -1}),
            (
                (
                    False,
                    True,
                ),
                {"axis": -1},
            ),
            (
                (
                    True,
                    True,
                ),
                {"axis": -1},
            ),
            (
                (
                    False,
                    False,
                    True,
                ),
                {"axis": -1},
            ),
            (
                (
                    True,
                    False,
                    True,
                ),
                {"axis": -1},
            ),
            (
                (
                    False,
                    True,
                    True,
                ),
                {"axis": -1},
            ),
            (
                (
                    True,
                    True,
                    True,
                ),
                {"axis": -1},
            ),
        ]
        self.op_class = Unique

    def test_op(self):
        if self.expect_success:
            for args, kwargs in self.ops_pars:
                op = self.op_class(*args, **kwargs)
                assert isinstance(op, self.op_class)
        else:
            for args, kwargs in self.ops_pars:

                def func():
                    return self.op_class(*args, **kwargs)

                with pytest.raises(RuntimeError):
                    func()

    def test_basic_vector(self):
        if not self.expect_success:
            raise pytest.skip("Requires numpy >= 1.13")
        # Basic test for a vector.
        # Done by using the op and checking that it returns the right
        # answer.

        x = theano.tensor.vector()
        ops = [self.op_class(*args, **kwargs) for args, kwargs in self.ops_pars]
        inp = np.asarray([2, 1, 3, 2], dtype=config.floatX)
        list_outs_expected = [
            [np.unique(inp, **kwargs)]
            if len(args) == 0
            else np.unique(inp, *args, **kwargs)
            for args, kwargs in self.ops_pars
        ]
        for op, outs_expected in zip(ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_basic_matrix(self):
        if not self.expect_success:
            raise pytest.skip("Requires numpy >= 1.13")
        # Basic test for a matrix.
        # Done by using the op and checking that it returns the right
        # answer.

        x = theano.tensor.matrix()
        ops = [self.op_class(*args, **kwargs) for args, kwargs in self.ops_pars]
        inp = np.asarray([[2, 1], [3, 2], [2, 1]], dtype=config.floatX)
        list_outs_expected = [
            [np.unique(inp, **kwargs)]
            if len(args) == 0
            else np.unique(inp, *args, **kwargs)
            for args, kwargs in self.ops_pars
        ]
        for op, outs_expected in zip(ops, list_outs_expected):
            f = theano.function(inputs=[x], outputs=op(x, return_list=True))
            outs = f(inp)
            # Compare the result computed to the expected value.
            for out, out_exp in zip(outs, outs_expected):
                utt.assert_allclose(out, out_exp)

    def test_infer_shape_vector(self):
        if not self.expect_success:
            raise pytest.skip("Requires numpy >= 1.13")
        # Testing the infer_shape with a vector.

        x = theano.tensor.vector()

        ops = [self.op_class(*args, **kwargs) for args, kwargs in self.ops_pars]
        for op in ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check(
                [x],
                [f],
                [np.asarray(np.array([2, 1, 3, 2]), dtype=config.floatX)],
                self.op_class,
            )

    def test_infer_shape_matrix(self):
        if not self.expect_success:
            raise pytest.skip("Requires numpy >= 1.13")
        # Testing the infer_shape with a matrix.

        x = theano.tensor.matrix()

        ops = [self.op_class(*args, **kwargs) for args, kwargs in self.ops_pars]
        for op in ops:
            if not op.return_inverse:
                continue
            if op.return_index:
                f = op(x)[2]
            else:
                f = op(x)[1]
            self._compile_and_check(
                [x],
                [f],
                [np.asarray(np.array([[2, 1], [3, 2], [2, 1]]), dtype=config.floatX)],
                self.op_class,
            )


class TestUnravelIndex(utt.InferShapeTester):
    def test_unravel_index(self):
        def check(shape, index_ndim, order):
            indices = np.arange(np.product(shape))
            # test with scalars and higher-dimensional indices
            if index_ndim == 0:
                indices = indices[-1]
            elif index_ndim == 2:
                indices = indices[:, np.newaxis]
            indices_symb = theano.shared(indices)

            # reference result
            ref = np.unravel_index(indices, shape, order=order)

            def fn(i, d):
                return function([], unravel_index(i, d, order=order))

            # shape given as a tuple
            f_array_tuple = fn(indices, shape)
            f_symb_tuple = fn(indices_symb, shape)
            np.testing.assert_equal(ref, f_array_tuple())
            np.testing.assert_equal(ref, f_symb_tuple())

            # shape given as an array
            shape_array = np.array(shape)
            f_array_array = fn(indices, shape_array)
            np.testing.assert_equal(ref, f_array_array())

            # shape given as a theano variable
            shape_symb = theano.shared(shape_array)
            f_array_symb = fn(indices, shape_symb)
            np.testing.assert_equal(ref, f_array_symb())

            # shape given as a Shape op (unravel_index will use get_vector_length
            # to infer the number of dimensions)
            indexed_array = theano.shared(np.random.uniform(size=shape_array))
            f_array_shape = fn(indices, indexed_array.shape)
            np.testing.assert_equal(ref, f_array_shape())

            # shape testing
            self._compile_and_check(
                [],
                unravel_index(indices, shape_symb, order=order),
                [],
                UnravelIndex,
            )

        for order in ("C", "F"):
            for index_ndim in (0, 1, 2):
                check((3,), index_ndim, order)
                check((3, 4), index_ndim, order)
                check((3, 4, 5), index_ndim, order)

        # must specify ndim if length of dims is not fixed
        with pytest.raises(ValueError):
            unravel_index(theano.tensor.ivector(), theano.tensor.ivector())

        # must provide integers
        with pytest.raises(TypeError):
            unravel_index(theano.tensor.fvector(), (3, 4))
        with pytest.raises(TypeError):
            unravel_index((3, 4), (3.4, 3.2))

        # dims must be a 1D sequence
        with pytest.raises(TypeError):
            unravel_index((3, 4), 3)
        with pytest.raises(TypeError):
            unravel_index((3, 4), ((3, 4),))


class TestRavelMultiIndex(utt.InferShapeTester):
    def test_ravel_multi_index(self):
        def check(shape, index_ndim, mode, order):
            multi_index = np.unravel_index(
                np.arange(np.product(shape)), shape, order=order
            )
            # create some invalid indices to test the mode
            if mode in ("wrap", "clip"):
                multi_index = (multi_index[0] - 1,) + multi_index[1:]
            # test with scalars and higher-dimensional indices
            if index_ndim == 0:
                multi_index = tuple(i[-1] for i in multi_index)
            elif index_ndim == 2:
                multi_index = tuple(i[:, np.newaxis] for i in multi_index)
            multi_index_symb = [theano.shared(i) for i in multi_index]

            # reference result
            ref = np.ravel_multi_index(multi_index, shape, mode, order)

            def fn(mi, s):
                return function([], ravel_multi_index(mi, s, mode, order))

            # shape given as a tuple
            f_array_tuple = fn(multi_index, shape)
            f_symb_tuple = fn(multi_index_symb, shape)
            np.testing.assert_equal(ref, f_array_tuple())
            np.testing.assert_equal(ref, f_symb_tuple())

            # shape given as an array
            shape_array = np.array(shape)
            f_array_array = fn(multi_index, shape_array)
            np.testing.assert_equal(ref, f_array_array())

            # shape given as a theano variable
            shape_symb = theano.shared(shape_array)
            f_array_symb = fn(multi_index, shape_symb)
            np.testing.assert_equal(ref, f_array_symb())

            # shape testing
            self._compile_and_check(
                [],
                [ravel_multi_index(multi_index, shape_symb, mode, order)],
                [],
                RavelMultiIndex,
            )

        for mode in ("raise", "wrap", "clip"):
            for order in ("C", "F"):
                for index_ndim in (0, 1, 2):
                    check((3,), index_ndim, mode, order)
                    check((3, 4), index_ndim, mode, order)
                    check((3, 4, 5), index_ndim, mode, order)

        # must provide integers
        with pytest.raises(TypeError):
            ravel_multi_index(
                (theano.tensor.fvector(), theano.tensor.ivector()), (3, 4)
            )
        with pytest.raises(TypeError):
            ravel_multi_index(((3, 4), theano.tensor.ivector()), (3.4, 3.2))

        # dims must be a 1D sequence
        with pytest.raises(TypeError):
            ravel_multi_index(((3, 4),), ((3, 4),))


def test_broadcast_shape():
    def shape_tuple(x, use_bcast=True):
        if use_bcast:
            return tuple(
                s if not bcast else 1
                for s, bcast in zip(tuple(x.shape), x.broadcastable)
            )
        else:
            return tuple(s for s in tuple(x.shape))

    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])
    b = np.broadcast(x, y)
    x_tt = tt.as_tensor_variable(x)
    y_tt = tt.as_tensor_variable(y)
    b_tt = broadcast_shape(x_tt, y_tt)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    # Now, we try again using shapes as the inputs
    #
    # This case also confirms that a broadcast dimension will
    # broadcast against a non-broadcast dimension when they're
    # both symbolic (i.e. we couldn't obtain constant values).
    b_tt = broadcast_shape(
        shape_tuple(x_tt, use_bcast=False),
        shape_tuple(y_tt, use_bcast=False),
        arrays_are_shapes=True,
    )
    assert any(
        isinstance(node.op, tt.opt.Assert)
        for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    )
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    b_tt = broadcast_shape(shape_tuple(x_tt), shape_tuple(y_tt), arrays_are_shapes=True)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    # These are all constants, so there shouldn't be any asserts in the
    # resulting graph.
    assert not any(
        isinstance(node.op, tt.opt.Assert)
        for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    )

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    b = np.broadcast(x, y)
    x_tt = tt.as_tensor_variable(x)
    y_tt = tt.as_tensor_variable(y)
    b_tt = broadcast_shape(x_tt, y_tt)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    b_tt = broadcast_shape(shape_tuple(x_tt), shape_tuple(y_tt), arrays_are_shapes=True)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    # TODO: This will work when/if we use a more sophisticated `is_same_graph`
    # implementation.
    # assert not any(
    #     isinstance(node.op, tt.opt.Assert)
    #     for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    # )

    x = np.empty((1, 2, 3))
    y = np.array(1)
    b = np.broadcast(x, y)
    x_tt = tt.as_tensor_variable(x)
    y_tt = tt.as_tensor_variable(y)
    b_tt = broadcast_shape(x_tt, y_tt)
    assert b_tt[0].value == 1
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    assert not any(
        isinstance(node.op, tt.opt.Assert)
        for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    )
    b_tt = broadcast_shape(shape_tuple(x_tt), shape_tuple(y_tt), arrays_are_shapes=True)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)

    x = np.empty((2, 1, 3))
    y = np.empty((2, 1, 1))
    b = np.broadcast(x, y)
    x_tt = tt.as_tensor_variable(x)
    y_tt = tt.as_tensor_variable(y)
    b_tt = broadcast_shape(x_tt, y_tt)
    assert b_tt[1].value == 1
    assert np.array_equal([z.eval() for z in b_tt], b.shape)
    # TODO: This will work when/if we use a more sophisticated `is_same_graph`
    # implementation.
    # assert not any(
    #     isinstance(node.op, tt.opt.Assert)
    #     for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    # )
    b_tt = broadcast_shape(shape_tuple(x_tt), shape_tuple(y_tt), arrays_are_shapes=True)
    assert np.array_equal([z.eval() for z in b_tt], b.shape)

    x1_shp_tt = tt.iscalar("x1")
    x2_shp_tt = tt.iscalar("x2")
    y1_shp_tt = tt.iscalar("y1")
    x_shapes = (1, x1_shp_tt, x2_shp_tt)
    x_tt = tt.ones(x_shapes)
    y_shapes = (y1_shp_tt, 1, x2_shp_tt)
    y_tt = tt.ones(y_shapes)
    b_tt = broadcast_shape(x_tt, y_tt)
    # TODO: This will work when/if we use a more sophisticated `is_same_graph`
    # implementation.
    # assert not any(
    #     isinstance(node.op, tt.opt.Assert)
    #     for node in tt.gof.graph.ops([x_tt, y_tt], b_tt)
    # )
    res = tt.as_tensor(b_tt).eval(
        {
            x1_shp_tt: 10,
            x2_shp_tt: 4,
            y1_shp_tt: 2,
        }
    )
    assert np.array_equal(res, (2, 10, 4))

    y_shapes = (y1_shp_tt, 1, y1_shp_tt)
    y_tt = tt.ones(y_shapes)
    b_tt = broadcast_shape(x_tt, y_tt)
    assert isinstance(b_tt[-1].owner.op, tt.opt.Assert)


class TestBroadcastTo(utt.InferShapeTester):

    rng = np.random.RandomState(43)

    def setup_method(self):
        super().setup_method()
        self.op_class = BroadcastTo
        self.op = broadcast_to

    @change_flags(compute_test_value="raise")
    def test_perform(self):
        a = tt.scalar()
        a.tag.test_value = 5

        s_1 = tt.iscalar("s_1")
        s_1.tag.test_value = 4
        shape = (s_1, 1)

        bcast_res = broadcast_to(a, shape)

        assert bcast_res.broadcastable == (False, True)

        bcast_np = np.broadcast_to(5, (4, 1))
        bcast_tt = bcast_res.get_test_value()

        assert np.array_equal(bcast_tt, bcast_np)
        assert np.shares_memory(bcast_tt, a.get_test_value())

    @pytest.mark.parametrize(
        "fn,input_dims",
        [
            [lambda x: broadcast_to(x, (1,)), (1,)],
            [lambda x: broadcast_to(x, (6, 2, 5, 3)), (1,)],
            [lambda x: broadcast_to(x, (6, 2, 5, 3)), (5, 1)],
            [lambda x: broadcast_to(x, (6, 2, 1, 3)), (2, 1, 3)],
        ],
    )
    def test_gradient(self, fn, input_dims):
        utt.verify_grad(
            fn,
            [np.random.rand(*input_dims).astype(config.floatX)],
            n_tests=1,
            rng=self.rng,
        )

    def test_infer_shape(self):
        a = tt.tensor(config.floatX, [False, True, False])
        shape = list(a.shape)
        out = self.op(a, shape)

        self._compile_and_check(
            [a] + shape,
            [out],
            [np.random.rand(2, 1, 3).astype(config.floatX), 2, 1, 3],
            self.op_class,
        )

        a = tt.tensor(config.floatX, [False, True, False])
        shape = [tt.iscalar() for i in range(4)]
        self._compile_and_check(
            [a] + shape,
            [self.op(a, shape)],
            [np.random.rand(2, 1, 3).astype(config.floatX), 6, 2, 5, 3],
            self.op_class,
        )

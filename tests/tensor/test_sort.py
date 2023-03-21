from functools import reduce
from itertools import chain, product

import numpy as np
import pytest

import aesara
from aesara.compile.mode import Mode
from aesara.tensor.sort import (
    ArgSortOp,
    SortOp,
    TopKOp,
    argsort,
    argtopk,
    sort,
    topk,
    topk_and_argtopk,
)
from aesara.tensor.type import (
    dmatrix,
    dvector,
    float_dtypes,
    integer_dtypes,
    lscalar,
    matrix,
    scalar,
    tensor,
    vector,
)
from tests import unittest_tools as utt


_all_dtypes = integer_dtypes + float_dtypes


def gen_unique_vector(size, dtype):
    rng = np.random.default_rng(utt.fetch_seed())
    # generate a randomized vector with unique elements
    retval = np.arange(size) * 3.0 + rng.uniform(-1.0, 1.0)
    return (retval[rng.permutation(size)] - size * 1.5).astype(dtype)


class TestSort:
    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())
        self.m_val = self.rng.random((3, 2))
        self.v_val = self.rng.random(4)

    def test1(self):
        a = dmatrix()
        w = sort(a)
        f = aesara.function([a], w)
        utt.assert_allclose(f(self.m_val), np.sort(self.m_val))

    def test2(self):
        a = dmatrix()
        axis = scalar()
        w = sort(a, axis)
        f = aesara.function([a, axis], w)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

    def test3(self):
        a = dvector()
        w2 = sort(a)
        f = aesara.function([a], w2)
        gv = f(self.v_val)
        gt = np.sort(self.v_val)
        utt.assert_allclose(gv, gt)

    def test4(self):
        a = dmatrix()
        axis = scalar()
        l = sort(a, axis, "mergesort")
        f = aesara.function([a, axis], l)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

    def test5(self):
        a1 = SortOp("mergesort", [])
        a2 = SortOp("quicksort", [])

        # All the below should give true
        assert a1 != a2
        assert a1 == SortOp("mergesort", [])
        assert a2 == SortOp("quicksort", [])

    def test_None(self):
        a = dmatrix()
        l = sort(a, None)
        f = aesara.function([a], l)
        gv = f(self.m_val)
        gt = np.sort(self.m_val, None)
        utt.assert_allclose(gv, gt)

    def test_grad_vector(self):
        data = self.rng.random(10).astype(aesara.config.floatX)
        utt.verify_grad(sort, [data])

    def test_grad_none_axis(self):
        data = self.rng.random(10).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        utt.verify_grad(lambda x: sort(x, 0), [data])

        data = self.rng.random((2, 3)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])

    def test_grad_negative_axis_2d(self):
        data = self.rng.random((2, 3)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])

    def test_grad_negative_axis_3d(self):
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])

    def test_grad_negative_axis_4d(self):
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, -4), [data])

    def test_grad_nonnegative_axis_2d(self):
        data = self.rng.random((2, 3)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])

    def test_grad_nonnegative_axis_3d(self):
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = self.rng.random((2, 3, 4)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])

    def test_grad_nonnegative_axis_4d(self):
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(aesara.config.floatX)
        utt.verify_grad(lambda x: sort(x, 3), [data])


class TestSortInferShape(utt.InferShapeTester):
    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())
        super().setup_method()

    def test_sort(self):
        x = matrix()
        self._compile_and_check(
            [x],
            [sort(x)],
            [self.rng.standard_normal(size=(10, 40)).astype(aesara.config.floatX)],
            SortOp,
        )
        self._compile_and_check(
            [x],
            [sort(x, axis=None)],
            [self.rng.standard_normal(size=(10, 40)).astype(aesara.config.floatX)],
            SortOp,
        )


def test_argsort():
    # Set up
    rng = np.random.default_rng(seed=utt.fetch_seed())
    m_val = rng.random((3, 2))
    v_val = rng.random(4)

    # Example 1
    a = dmatrix()
    w = argsort(a)
    f = aesara.function([a], w)
    gv = f(m_val)
    gt = np.argsort(m_val)
    utt.assert_allclose(gv, gt)

    # Example 2
    a = dmatrix()
    axis = lscalar()
    w = argsort(a, axis)
    f = aesara.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

    # Example 3
    a = dvector()
    w2 = argsort(a)
    f = aesara.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    utt.assert_allclose(gv, gt)

    # Example 4
    a = dmatrix()
    axis = lscalar()
    l = argsort(a, axis, "mergesort")
    f = aesara.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

    # Example 5
    a = dmatrix()
    axis = lscalar()
    a1 = ArgSortOp("mergesort", [])
    a2 = ArgSortOp("quicksort", [])
    # All the below should give true
    assert a1 != a2
    assert a1 == ArgSortOp("mergesort", [])
    assert a2 == ArgSortOp("quicksort", [])

    # Example 6: Testing axis=None
    a = dmatrix()
    w2 = argsort(a, None)
    f = aesara.function([a], w2)
    gv = f(m_val)
    gt = np.argsort(m_val, None)
    utt.assert_allclose(gv, gt)


def test_argsort_grad():
    rng = np.random.default_rng(seed=utt.fetch_seed())
    # Testing grad of argsort
    data = rng.random((2, 3)).astype(aesara.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-1), [data])

    data = rng.random((2, 3, 4, 5)).astype(aesara.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-3), [data])

    data = rng.random((2, 3, 3)).astype(aesara.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=2), [data])


class TestTopK:
    mode = None
    op_class = TopKOp

    def setup_method(self):
        pass

    @pytest.mark.parametrize("dtype", _all_dtypes)
    @pytest.mark.parametrize("idx_dtype", integer_dtypes)
    @pytest.mark.parametrize("axis", [-1, 0, None])
    @pytest.mark.parametrize("sorted", [False])
    def test_argtopk_sanity(self, dtype, idx_dtype, axis, sorted):
        x = vector(name="x", dtype=dtype)
        fn = aesara.function(
            [x],
            argtopk(x, 1, axis=axis, sorted=sorted, idx_dtype=idx_dtype),
            mode=self.mode,
        )
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == np.asarray([0], dtype=idx_dtype)
        assert yval.dtype == np.dtype(idx_dtype)

    @pytest.mark.parametrize("dtype", _all_dtypes)
    @pytest.mark.parametrize("axis", [-1, 0, None])
    @pytest.mark.parametrize("sorted", [False])
    def test_topk_sanity(self, dtype, axis, sorted):
        x = vector(name="x", dtype=dtype)
        fn = aesara.function([x], topk(x, 1, axis=axis, sorted=sorted), mode=self.mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        xval = np.asarray([1]).astype(dtype)
        yval = fn(xval)
        assert yval == xval
        assert yval.dtype == xval.dtype

    @pytest.mark.parametrize("dtype", _all_dtypes)
    @pytest.mark.parametrize("idx_dtype", integer_dtypes)
    @pytest.mark.parametrize("axis", [-1, 0, None])
    @pytest.mark.parametrize("sorted", [False])
    def test_combined_sanity(self, dtype, idx_dtype, axis, sorted):
        x = vector(name="x", dtype=dtype)
        yv, yi = topk_and_argtopk(x, 1, axis=axis, sorted=sorted, idx_dtype=idx_dtype)
        fn = aesara.function([x], [yv, yi], mode=self.mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        xval = np.asarray([1]).astype(dtype)
        yvval, yival = fn(xval)
        assert yival == np.asarray([0], dtype=idx_dtype)
        utt.assert_allclose(xval, yvval)
        assert yvval.dtype == xval.dtype
        assert yival.dtype == np.dtype(idx_dtype)

    @pytest.mark.parametrize(
        "size, k, dtype, sorted",
        chain(
            product(
                (16, 61, 257),
                (1, -1, -10, "n//2", "n-1", "-n", "1-n"),
                ("float64", "float16", "int16", "int8"),
                (False,),
            ),
            ((2049, 1337, "float64", False),),
        ),
    )
    def test_topk_1d(self, size, k, dtype, sorted):
        if isinstance(k, str):
            k = eval(k.replace("n", str(size)))

        x = vector(name="x", dtype=dtype)
        y = topk(x, k, sorted=sorted)
        fn = aesara.function([x], y, mode=self.mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        # assert local_useless_topk opt is done properly
        assert 1 == len(fn.maker.fgraph.outputs[0].owner.outputs)

        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.sort(xval)[idx]

        assert yval.dtype == goal.dtype
        utt.assert_allclose(goal, np.sort(yval))

    @pytest.mark.parametrize(
        "size, k, dtype, sorted, idx_dtype",
        chain(
            product(
                (16, 61, 257),
                (1, -1, -10, "n//2", "n-1", "-n"),
                ("float32", "int32"),
                (False,),
                ("int32", "int64"),
            ),
            ((2049, 1337, "float32", False, "int32"),),
        ),
    )
    def test_argtopk_1d(self, size, k, dtype, sorted, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace("n", str(size)))

        x = vector(name="x", dtype=dtype)
        y = argtopk(x, k, sorted=sorted, idx_dtype=idx_dtype)
        fn = aesara.function([x], y, mode=self.mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)

        # assert local_useless_topk opt is done properly
        assert 1 == len(fn.maker.fgraph.outputs[0].owner.outputs)

        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.argsort(xval)[idx].astype(idx_dtype)

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yval)] == xval[np.sort(goal)])

    @pytest.mark.parametrize(
        "size, k, dtype, sorted, idx_dtype",
        chain(
            product(
                (16, 61, 257),
                (1, -1, 10, "n//2", "n-1", "1-n"),
                ("float32", "int32"),
                (False,),
                ("int32", "int64"),
            ),
            ((2049, 1337, "float32", False, "int32"),),
        ),
    )
    def test_combined_1d(self, size, k, dtype, sorted, idx_dtype):
        if isinstance(k, str):
            k = eval(k.replace("n", str(size)))

        x = vector(name="x", dtype=dtype)
        yv, yi = topk_and_argtopk(x, k, sorted=sorted, idx_dtype=idx_dtype)
        fn = aesara.function([x], [yv, yi], mode=self.mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        # generate a all-unique array
        xval = gen_unique_vector(size, dtype)
        yvval, yival = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goali = np.argsort(xval)[idx].astype(idx_dtype)
        goalv = xval[goali]

        # due to uniqueness, we expect indices same
        assert np.all(xval[np.sort(yival)] == xval[np.sort(goali)])
        utt.assert_allclose(np.sort(yvval), goalv)

    @pytest.mark.parametrize(
        "size, k, dtype, sorted",
        chain(
            product((18, 62, 258), (1, -1, "n//2"), ("int32", "float32"), (False,)),
            ((2048, 1337, "float32", False),),
        ),
    )
    def test_argtopk_1d_collision(self, size, k, dtype, sorted):
        # with non-unique kth max value
        if isinstance(k, str):
            k = eval(k.replace("n", str(size)))

        x = vector(name="x", dtype=dtype)
        y = argtopk(x, k, sorted=sorted, idx_dtype="int32")
        # DebugMode won't like the index change on collision on CPU
        # So don't use DebugMode here.
        mode = self.mode
        if isinstance(self.mode, aesara.compile.debugmode.DebugMode):
            mode = Mode(optimizer=mode.optimizer)
        fn = aesara.function([x], y, mode=mode)
        assert any(isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes)
        rng = np.random.default_rng(utt.fetch_seed())
        xval = np.repeat(rng.uniform(-100.0, 100.0, size=size // 2).astype(dtype), 2)
        xval = xval[rng.permutation(size)]
        yval = fn(xval)
        idx = slice(-k, None) if k > 0 else slice(-k)
        goal = np.argsort(xval)[idx].astype("int32")
        utt.assert_allclose(np.sort(xval[yval]), np.sort(xval[goal]))

    @pytest.mark.parametrize(
        "shp, k_, dtype, sorted, idx_dtype",
        product(
            (
                (17, 15),
                (2, 3, 5, 7, 11),
                (500, 5, 3),
            ),  # NB: Test may fail with bigger sizes (e.g. (2017, 5, 3)) due to "too many resources requested" kernel error on some GPUs.
            (-1, "(1+n)//2", "-n", "1-n"),
            ("float32", "int32"),
            (False,),
            ("int32", "int64"),
        ),
    )
    def test_argtopk_nd(self, shp, k_, dtype, sorted, idx_dtype):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace("n", str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            x = tensor(name="x", shape=(None,) * len(shp), dtype=dtype)
            y = argtopk(x, k, axis=axis, sorted=sorted, idx_dtype=idx_dtype)
            fn = aesara.function([x], y, mode=self.mode)
            assert any(
                isinstance(n.op, self.op_class) for n in fn.maker.fgraph.apply_nodes
            )
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, dtype).reshape(shp)
            yval = fn(xval)
            idx = slice(-k, None) if k > 0 else slice(-k)
            l = axis % ndim
            r = ndim - l
            idx = (slice(None),) * l + (idx,) + (slice(None),) * (r - 1)
            goal = np.argsort(xval, axis=axis)[idx].astype(idx_dtype)

            assert np.all(np.sort(yval, axis=axis) == np.sort(goal, axis=axis))

    @pytest.mark.parametrize("shp", ((257,), (17, 15), (5, 3, 5, 3), (2, 3, 5, 7, 11)))
    @pytest.mark.parametrize("k_", (1, -1, "(1+n)//2", "n-1", "-n", "1-n"))
    @pytest.mark.parametrize("sorted", [False])
    def test_grad(self, shp, k_, sorted):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace("n", str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            # make input away from undefined gradient (where some inputs are equal)
            xval = gen_unique_vector(
                reduce(int.__mul__, shp), dtype=aesara.config.floatX
            ).reshape(shp)
            utt.verify_grad(
                lambda x: topk(x, k, axis=axis, sorted=sorted), [xval], eps=1e-2
            )


class TestTopKInferShape(utt.InferShapeTester):
    @pytest.mark.parametrize(
        "shp", ((2, 3), (15, 17), (11, 7, 5), (2, 3, 5, 7, 11), (2, 4, 3, 1))
    )
    @pytest.mark.parametrize("k_", (1, "(1+n)//2", "n-1", "n"))
    def test_combined_infer_shape(self, shp, k_):
        ndim = len(shp)
        for axis in range(-ndim, ndim):
            if isinstance(k_, str):
                k = eval(k_.replace("n", str(shp[axis])))
            else:
                k = k_

            if k == 0:
                continue

            x = tensor(name="x", shape=(None,) * len(shp), dtype=aesara.config.floatX)
            yv, yi = topk_and_argtopk(x, k, axis=axis, sorted=False, idx_dtype="int32")
            size = reduce(int.__mul__, shp)
            xval = gen_unique_vector(size, aesara.config.floatX).reshape(shp)
            self._compile_and_check([x], [yv, yi], [xval], TopKOp)

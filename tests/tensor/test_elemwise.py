import math
from copy import copy

import numpy as np
import pytest

import tests.unittest_tools as utt
import theano
import theano.scalar as ts
from tests import unittest_tools
from tests.tensor.test_math import reduce_bitwise_and
from theano.compile.mode import Mode
from theano.configdefaults import config
from theano.graph.basic import Variable
from theano.graph.fg import FunctionGraph
from theano.link.basic import PerformLinker
from theano.link.c.basic import CLinker, OpWiseCLinker
from theano.tensor import as_tensor_variable
from theano.tensor.basic import second
from theano.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from theano.tensor.math import all as tt_all
from theano.tensor.math import any as tt_any
from theano.tensor.type import (
    TensorType,
    bmatrix,
    bscalar,
    discrete_dtypes,
    matrix,
    scalar,
    vectors,
)


class TestDimShuffle(unittest_tools.InferShapeTester):
    op = DimShuffle
    type = TensorType
    dtype = theano.config.floatX

    def with_linker(self, linker):
        for xsh, shuffle, zsh in [
            ((2, 3), (1, "x", 0), (3, 1, 2)),
            ((1, 2, 3), (1, 2), (2, 3)),
            ((1, 2, 1, 3), (1, 3), (2, 3)),
            ((2, 3, 4), (2, 1, 0), (4, 3, 2)),
            ((2, 3, 4), ("x", 2, 1, 0, "x"), (1, 4, 3, 2, 1)),
            ((1, 4, 3, 2, 1), (3, 2, 1), (2, 3, 4)),
            ((1, 1, 4), (1, 2), (1, 4)),
            ((1, 1, 1), (), ()),
            ((1,), ("x", "x"), (1, 1)),
        ]:
            ib = [(entry == 1) for entry in xsh]
            x = self.type(self.dtype, ib)("x")
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x], [e])).make_function()
            assert f(np.ones(xsh, dtype=self.dtype)).shape == zsh
            # test that DimShuffle.infer_shape work correctly
            x = self.type(self.dtype, ib)("x")
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x], [e.shape])).make_function()
            assert all(f(np.ones(xsh, dtype=self.dtype))) == all(zsh)

        # Test when we drop a axis that is not broadcastable
        ib = [False, True, False]
        x = self.type(self.dtype, ib)("x")
        with pytest.raises(ValueError):
            self.op(ib, shuffle)

        # Test when we drop a axis that don't have shape 1
        ib = [True, True, False]
        x = self.type(self.dtype, ib)("x")
        e = self.op(ib, (1, 2))(x)
        f = copy(linker).accept(FunctionGraph([x], [e.shape])).make_function()
        with pytest.raises(TypeError):
            f(np.ones((2, 1, 4)))

        # Test that we can't take a dimensions multiple time
        xsh, shuffle, zsh = ((1, 1, 4), (0, 1, 2, 0), (1, 4))
        ib = [False, True, False]
        x = self.type(self.dtype, ib)("x")
        with pytest.raises(ValueError):
            DimShuffle(ib, shuffle)

    def test_perform(self):
        self.with_linker(PerformLinker())

    def test_c_or_py(self):
        # Shape op don't have C code.
        # But This will test DimShuffle c code
        self.with_linker(OpWiseCLinker())

    def test_infer_shape(self):

        for xsh, shuffle in [
            ((2, 3), (1, "x", 0)),
            ((1, 2, 3), (1, 2)),
            ((1, 2, 1, 3), (1, 3)),
            ((2, 3, 4), (2, 1, 0)),
            ((2, 3, 4), ("x", 2, 1, 0, "x")),
            ((1, 4, 3, 2, 1), (3, 2, 1)),
            ((1, 1, 4), (1, 2)),
            ((1, 1, 1), ()),
            ((1,), ("x", "x")),
        ]:
            ib = [(entry == 1) for entry in xsh]
            adtens = self.type(self.dtype, ib)("x")
            adtens_val = np.ones(xsh, dtype=self.dtype)
            self._compile_and_check(
                [adtens],
                [self.op(ib, shuffle)(adtens)],
                [adtens_val],
                self.op,
                warn=False,
            )

    def test_too_big_rank(self):
        x = self.type(self.dtype, broadcastable=())()
        y = x.dimshuffle(("x",) * (np.MAXDIMS + 1))
        with pytest.raises(ValueError):
            y.eval({x: 0})


class TestBroadcast:
    # this is to allow other types to reuse this class to test their ops
    type = TensorType
    op = Elemwise

    ctype = TensorType
    cop = Elemwise

    openmp_minsize = 2 * config.openmp_elemwise_minsize
    openmp_minsize_sqrt = int(math.ceil(math.sqrt(openmp_minsize)))

    # The order is important if you change them.
    linkers = [PerformLinker, CLinker]

    def rand_val(self, shp):
        return np.asarray(np.random.rand(*shp), dtype=theano.config.floatX)

    def rand_cval(self, shp):
        return np.asarray(np.random.rand(*shp), dtype=theano.config.floatX)

    def setup_method(self):
        unittest_tools.seed_rng()

    def with_linker(self, linker, op, type, rand_val):
        for xsh, ysh in [
            ((3, 5), (3, 5)),
            ((3, 5), (1, 5)),
            ((3, 5), (3, 1)),
            ((1, 5), (5, 1)),
            ((1, 1), (1, 1)),
            ((self.openmp_minsize,), (self.openmp_minsize,)),
            (
                (self.openmp_minsize_sqrt, self.openmp_minsize_sqrt),
                (self.openmp_minsize_sqrt, self.openmp_minsize_sqrt),
            ),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (1, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 1, 1, 1)),
            ((), ()),
        ]:
            x = type(theano.config.floatX, [(entry == 1) for entry in xsh])("x")
            y = type(theano.config.floatX, [(entry == 1) for entry in ysh])("y")
            e = op(ts.add)(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            unittest_tools.assert_allclose(f(xv, yv), zv)

            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, PerformLinker):
                x = type(theano.config.floatX, [(entry == 1) for entry in xsh])("x")
                y = type(theano.config.floatX, [(entry == 1) for entry in ysh])("y")
                e = op(ts.add)(x, y)
                f = (
                    copy(linker)
                    .accept(FunctionGraph([x, y], [e.shape]))
                    .make_function()
                )
                assert tuple(f(xv, yv)) == tuple(zv.shape)

    def with_linker_inplace(self, linker, op, type, rand_val):
        for xsh, ysh in [
            ((5, 5), (5, 5)),
            ((5, 5), (1, 5)),
            ((5, 5), (5, 1)),
            ((1, 1), (1, 1)),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (1, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 1, 1, 1)),
            ((), ()),
        ]:
            x = type(theano.config.floatX, [(entry == 1) for entry in xsh])("x")
            y = type(theano.config.floatX, [(entry == 1) for entry in ysh])("y")
            e = op(ts.Add(ts.transfer_type(0)), {0: 0})(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            f(xv, yv)

            assert (xv == zv).all()
            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, PerformLinker):
                x = type(theano.config.floatX, [(entry == 1) for entry in xsh])("x")
                y = type(theano.config.floatX, [(entry == 1) for entry in ysh])("y")
                e = op(ts.Add(ts.transfer_type(0)), {0: 0})(x, y)
                f = (
                    copy(linker)
                    .accept(FunctionGraph([x, y], [e.shape]))
                    .make_function()
                )
                xv = rand_val(xsh)
                yv = rand_val(ysh)
                zv = xv + yv

                f(xv, yv)

                assert xv.shape == zv.shape

    def test_perform(self):
        self.with_linker(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        self.with_linker(CLinker(), self.cop, self.ctype, self.rand_cval)

    def test_perform_inplace(self):
        self.with_linker_inplace(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_inplace(self):
        self.with_linker_inplace(CLinker(), self.cop, self.ctype, self.rand_cval)

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_fill(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(theano.config.floatX, [0, 0])("x")
            y = t(theano.config.floatX, [1, 1])("y")
            e = op(ts.Second(ts.transfer_type(0)), {0: 0})(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((5, 5))
            yv = rval((1, 1))
            f(xv, yv)
            assert (xv == yv).all()

    def test_fill_var(self):
        x = matrix()
        x.fill(3)

    def test_fill_grad(self):
        # Fix bug reported at
        # https://groups.google.com/d/topic/theano-users/nQshB8gUA6k/discussion
        x = TensorType(config.floatX, [0, 1, 0])("x")
        y = TensorType(config.floatX, [0, 1, 0])("y")
        e = second(x, y)
        theano.grad(e.sum(), y)

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_weird_strides(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(theano.config.floatX, [0, 0, 0, 0, 0])("x")
            y = t(theano.config.floatX, [0, 0, 0, 0, 0])("y")
            e = op(ts.add)(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((2, 2, 2, 2, 2))
            yv = rval((2, 2, 2, 2, 2)).transpose(4, 0, 3, 1, 2)
            zv = xv + yv
            assert (f(xv, yv) == zv).all()

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_same_inputs(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(theano.config.floatX, [0, 0])("x")
            e = op(ts.add)(x, x)
            f = linker().accept(FunctionGraph([x], [e])).make_function()
            xv = rval((2, 2))
            zv = xv + xv
            assert (f(xv) == zv).all()


class TestCAReduce(unittest_tools.InferShapeTester):
    op = CAReduce
    cases = [
        ((5, 6), None),
        ((5, 6), (0, 1)),
        ((5, 6), (0,)),
        ((5, 6), (1,)),
        ((5, 6), (-1,)),
        ((5, 6), (-2,)),
        ((5, 6), ()),
        ((2, 3, 4, 5), (0, 1, 3)),
        ((2, 3, 4, 5), (-2, -3)),
        ((5, 0), None),
        ((5, 0), (0,)),
        ((5, 0), (1,)),
        ((5, 0), ()),
        ((), None),
        ((), ()),
    ]
    type = TensorType

    def with_mode(
        self,
        mode,
        scalar_op=ts.add,
        dtype="floatX",
        pre_scalar_op=None,
        test_nan=False,
        tensor_op=None,
    ):
        for xsh, tosum in self.cases:
            if dtype == "floatX":
                dtype = theano.config.floatX
            x = self.type(dtype, [(entry == 1) for entry in xsh])("x")
            d = {}
            if pre_scalar_op is not None:
                d = {"pre_scalar_op": pre_scalar_op}
            if tensor_op is None:
                e = as_tensor_variable(self.op(scalar_op, axis=tosum, **d)(x))
            else:
                e = as_tensor_variable(tensor_op(x, axis=tosum, **d))

            if tosum is None:
                tosum = list(range(len(xsh)))

            f = theano.function([x], e, mode=mode)
            xv = np.asarray(np.random.rand(*xsh))

            if dtype not in discrete_dtypes:
                xv = np.asarray(xv, dtype=dtype)
            else:
                xv = np.asarray(xv < 0.5, dtype=dtype)

            if test_nan and xv.size > 0:
                if len(xsh) > 0:
                    xv = xv.flatten()
                    xv[0] = np.nan
                    xv = xv.reshape(*xsh)
                else:
                    xv = np.asarray(np.nan, dtype=dtype)
            zv = xv
            if pre_scalar_op is not None:
                zv = Elemwise(scalar_op=pre_scalar_op)(x).eval({x: xv})
            numpy_raised = False
            if len(tosum) > 1 and any([a < 0 for a in tosum]):
                # In that case, we need to use the good order of axis
                # in the reduction.
                axis2 = []
                for a in tosum:
                    if a < 0:
                        axis2.append(a + len(xsh))
                    else:
                        axis2.append(a)
                assert len(axis2) == len(tosum)
                tosum = tuple(axis2)
            if tensor_op == tt_all:
                for axis in reversed(sorted(tosum)):
                    zv = np.all(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif tensor_op == tt_any:
                for axis in reversed(sorted(tosum)):
                    zv = np.any(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif scalar_op == ts.add:
                for axis in reversed(sorted(tosum)):
                    zv = np.add.reduce(zv, axis)
                if dtype == "bool":
                    # np.add of a bool upcast, while CAReduce don't
                    zv = zv.astype(dtype)
            elif scalar_op == ts.mul:
                for axis in reversed(sorted(tosum)):
                    zv = np.multiply.reduce(zv, axis)
            elif scalar_op == ts.scalar_maximum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = np.maximum.reduce(zv, axis)
                except ValueError:
                    numpy_raised = True
            elif scalar_op == ts.scalar_minimum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = np.minimum.reduce(zv, axis)
                except ValueError:
                    numpy_raised = True
            elif scalar_op == ts.or_:
                for axis in reversed(sorted(tosum)):
                    zv = np.bitwise_or.reduce(zv, axis)
            elif scalar_op == ts.and_:
                for axis in reversed(sorted(tosum)):
                    zv = reduce_bitwise_and(zv, axis, dtype=dtype)
            elif scalar_op == ts.xor:
                # There is no identity value for the xor function
                # So we can't support shape of dimensions 0.
                if np.prod(zv.shape) == 0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = np.bitwise_xor.reduce(zv, axis)
            else:
                raise Exception(
                    f"Test for CAReduce with scalar_op {scalar_op} not implemented"
                )
            if scalar_op in [ts.scalar_maximum, ts.scalar_minimum] and numpy_raised:
                with pytest.raises(ValueError):
                    f(xv)
            else:
                if test_nan:
                    try:
                        assert self.type.values_eq(f(xv), zv), (f(xv), zv)
                    except NotImplementedError:
                        # GpuCAReduce don't implement all cases when size is 0
                        assert xv.size == 0
                else:
                    try:
                        f_xv = f(xv)
                        assert f_xv.shape == zv.shape, (f_xv, zv)
                        utt.assert_allclose(zv, f_xv)
                    except NotImplementedError:
                        # GpuCAReduce don't implement all cases when size is 0
                        assert xv.size == 0

            x = self.type(dtype, [(entry == 1) for entry in xsh])("x")
            if tensor_op is None:
                e = self.op(scalar_op, axis=tosum)(x)
            else:
                e = tensor_op(x, axis=tosum)
            if tosum is None:
                tosum = list(range(len(xsh)))
            f = theano.function([x], e.shape, mode=mode)
            if not (
                scalar_op in [ts.scalar_maximum, ts.scalar_minimum]
                and (xsh == () or np.prod(xsh) == 0)
            ):
                try:
                    assert all(f(xv) == zv.shape)
                except NotImplementedError:
                    # GpuCAReduce don't implement all cases when size is 0
                    assert xv.size == 0

    def test_perform_noopt(self):
        self.with_mode(Mode(linker="py", optimizer=None), ts.add, dtype="floatX")

    def test_perform(self):
        for dtype in ["bool", "floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_mode(Mode(linker="py"), ts.add, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.mul, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.scalar_maximum, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.scalar_minimum, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.and_, dtype=dtype, tensor_op=tt_all)
            self.with_mode(Mode(linker="py"), ts.or_, dtype=dtype, tensor_op=tt_any)
        for dtype in ["int8", "uint8"]:
            self.with_mode(Mode(linker="py"), ts.or_, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.and_, dtype=dtype)
            self.with_mode(Mode(linker="py"), ts.xor, dtype=dtype)

    def test_perform_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_mode(Mode(linker="py"), ts.add, dtype=dtype, test_nan=True)
            self.with_mode(Mode(linker="py"), ts.mul, dtype=dtype, test_nan=True)
            self.with_mode(
                Mode(linker="py"), ts.scalar_maximum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="py"), ts.scalar_minimum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="py"),
                ts.or_,
                dtype=dtype,
                test_nan=True,
                tensor_op=tt_any,
            )
            self.with_mode(
                Mode(linker="py"),
                ts.and_,
                dtype=dtype,
                test_nan=True,
                tensor_op=tt_all,
            )

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_noopt(self):
        # We need to make sure that we cover the corner cases that
        # optimizations normally cover
        self.with_mode(Mode(linker="c", optimizer=None), ts.add, dtype="floatX")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        for dtype in ["bool", "floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), ts.add, dtype=dtype)
            self.with_mode(Mode(linker="c"), ts.mul, dtype=dtype)
        for dtype in ["bool", "floatX", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), ts.scalar_minimum, dtype=dtype)
            self.with_mode(Mode(linker="c"), ts.scalar_maximum, dtype=dtype)
            self.with_mode(Mode(linker="c"), ts.and_, dtype=dtype, tensor_op=tt_all)
            self.with_mode(Mode(linker="c"), ts.or_, dtype=dtype, tensor_op=tt_any)
        for dtype in ["bool", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), ts.or_, dtype=dtype)
            self.with_mode(Mode(linker="c"), ts.and_, dtype=dtype)
            self.with_mode(Mode(linker="c"), ts.xor, dtype=dtype)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_mode(Mode(linker="c"), ts.add, dtype=dtype, test_nan=True)
            self.with_mode(Mode(linker="c"), ts.mul, dtype=dtype, test_nan=True)
        for dtype in ["floatX"]:
            self.with_mode(
                Mode(linker="c"), ts.scalar_minimum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="c"), ts.scalar_maximum, dtype=dtype, test_nan=True
            )

    def test_infer_shape(self, dtype=None, pre_scalar_op=None):
        if dtype is None:
            dtype = theano.config.floatX
        for xsh, tosum in self.cases:
            x = self.type(dtype, [(entry == 1) for entry in xsh])("x")
            if pre_scalar_op is not None:
                x = pre_scalar_op(x)
            if tosum is None:
                tosum = list(range(len(xsh)))
            xv = np.asarray(np.random.rand(*xsh), dtype=dtype)
            d = {}
            if pre_scalar_op is not None:
                xv = x.eval({x.owner.inputs[0]: xv})
                d = {pre_scalar_op: pre_scalar_op}
            self._compile_and_check(
                [x],
                [self.op(ts.add, axis=tosum, *d)(x)],
                [xv],
                self.op,
                ["local_cut_useless_reduce"],
                warn=0 not in xsh,
            )


class TestBitOpReduceGrad:
    def setup_method(self):
        self.rng = np.random.RandomState(unittest_tools.fetch_seed())

    def test_all_grad(self):
        x = bmatrix("x")
        x_all = x.all()
        gx = theano.grad(x_all, x)
        f = theano.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype("int8")
        for x_val in (x_random, np.zeros_like(x_random), np.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert np.all(gx_val == 0)

    def test_any_grad(self):
        x = bmatrix("x")
        x_all = x.any()
        gx = theano.grad(x_all, x)
        f = theano.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype("int8")
        for x_val in (x_random, np.zeros_like(x_random), np.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert np.all(gx_val == 0)


class TestElemwise(unittest_tools.InferShapeTester):
    def test_elemwise_grad_bool(self):
        x = scalar(dtype="bool")
        y = bscalar()
        z = x * y
        dx, dy = theano.grad(z, [x, y])

    def test_infer_shape(self):

        for s_left, s_right in [
            ((5, 6), (5, 6)),
            ((5, 6), (5, 1)),
            ((5, 6), (1, 6)),
            ((5, 1), (5, 6)),
            ((1, 6), (5, 6)),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (2, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 3, 4, 5)),
            ((2, 1, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 1), (2, 3, 4, 5)),
        ]:
            dtype = theano.config.floatX
            t_left = TensorType(dtype, [(entry == 1) for entry in s_left])()
            t_right = TensorType(dtype, [(entry == 1) for entry in s_right])()
            t_left_val = np.zeros(s_left, dtype=dtype)
            t_right_val = np.zeros(s_right, dtype=dtype)
            self._compile_and_check(
                [t_left, t_right],
                [Elemwise(ts.add)(t_left, t_right)],
                [t_left_val, t_right_val],
                Elemwise,
            )

    def test_input_dimensions_overflow(self):
        # Elemwise.perform used to compute the product
        # of input shapes to check if there was a zero in them,
        # it overflowed in this case.
        a, b, c, d, e, f = vectors("abcdef")
        s = a + b + c + d + e + f
        g = theano.function([a, b, c, d, e, f], s, mode=Mode(linker="py"))
        g(*[np.zeros(2 ** 11, config.floatX) for i in range(6)])


def test_not_implemented_elemwise_grad():
    # Regression test for unimplemented gradient in an Elemwise Op.

    class TestOp(ts.ScalarOp):
        def __init__(self):
            self.output_types_preference = ts.upgrade_to_float

        def impl(self, n, x):
            return x * n

        def grad(self, inputs, gout):
            (n, x) = inputs
            (gz,) = gout
            dy_dx = n
            return [theano.gradient.grad_not_implemented(self, 0, n), gz * dy_dx]

    test_op = Elemwise(TestOp())
    x = scalar()
    assert isinstance(theano.gradient.grad(test_op(2, x), x), Variable)

    # Verify that trying to use the not implemented gradient fails.
    with pytest.raises(theano.gradient.NullTypeGradError):
        theano.gradient.grad(test_op(x, 2), x)

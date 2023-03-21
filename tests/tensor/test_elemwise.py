import math
import re
import tracemalloc
from copy import copy

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import tests.unittest_tools as utt
from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Variable
from aesara.graph.fg import FunctionGraph
from aesara.link.basic import PerformLinker
from aesara.link.c.basic import CLinker, OpWiseCLinker
from aesara.tensor import as_tensor_variable
from aesara.tensor.basic import second
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.exceptions import ShapeError
from aesara.tensor.math import all as at_all
from aesara.tensor.math import any as at_any
from aesara.tensor.math import exp
from aesara.tensor.type import (
    TensorType,
    bmatrix,
    bscalar,
    discrete_dtypes,
    lscalar,
    matrix,
    scalar,
    tensor,
    vector,
    vectors,
)
from tests import unittest_tools
from tests.link.test_link import make_function
from tests.tensor.test_math import reduce_bitwise_and


class TestDimShuffle(unittest_tools.InferShapeTester):
    op = DimShuffle
    type = TensorType
    dtype = aesara.config.floatX

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
            i_shape = [entry if entry == 1 else None for entry in xsh]
            ib = [entry == 1 for entry in i_shape]
            x = self.type(self.dtype, shape=i_shape)("x")
            e = self.op(ib, shuffle)(x)
            f = aesara.function([x], e, mode=Mode(linker=linker))
            assert f(np.ones(xsh, dtype=self.dtype)).shape == zsh
            # test that DimShuffle.infer_shape work correctly
            x = self.type(self.dtype, shape=i_shape)("x")
            e = self.op(ib, shuffle)(x)
            f = aesara.function(
                [x], e.shape, mode=Mode(linker=linker), on_unused_input="ignore"
            )
            assert all(f(np.ones(xsh, dtype=self.dtype))) == all(zsh)

        # Test when we drop a axis that is not broadcastable
        ib = [False, True, False]
        x = self.type(self.dtype, shape=(None, 1, None))("x")
        with pytest.raises(ValueError):
            self.op(ib, shuffle)

        # Test when we drop a axis that don't have shape 1
        ib = [True, True, False]
        x = self.type(self.dtype, shape=(1, 1, None))("x")
        e = self.op(ib, (1, 2))(x)
        f = aesara.function([x], e.shape, mode=Mode(linker=linker))
        with pytest.raises(TypeError):
            f(np.ones((2, 1, 4)))

        # Test that we can't take a dimensions multiple time
        xsh, shuffle, zsh = ((1, 1, 4), (0, 1, 2, 0), (1, 4))
        ib = [False, True, False]
        x = self.type(self.dtype, shape=(None, 1, None))("x")
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
            i_shape = [entry if entry == 1 else None for entry in xsh]
            ib = [(entry == 1) for entry in xsh]
            adtens = self.type(self.dtype, shape=i_shape)("x")
            adtens_val = np.ones(xsh, dtype=self.dtype)
            self._compile_and_check(
                [adtens],
                [self.op(ib, shuffle)(adtens)],
                [adtens_val],
                self.op,
                warn=False,
            )

    def test_too_big_rank(self):
        x = self.type(self.dtype, shape=())()
        y = x.dimshuffle(("x",) * (np.MAXDIMS + 1))
        with pytest.raises(ValueError):
            y.eval({x: 0})

    def test_c_views(self):
        x_at = vector()
        thunk, inputs, outputs = (
            CLinker().accept(FunctionGraph([x_at], [x_at[None]])).make_thunk()
        )

        # This is a little hackish, but we're hoping that--by running this more than
        # a few times--we're more likely to run into random memory that isn't the same
        # as the broadcasted value; that way, we'll be able to tell that we're getting
        # junk data from a poorly constructed array view.
        x_val = np.broadcast_to(2039, (5000,))
        for i in range(1000):
            inputs[0].storage[0] = x_val
            thunk()
            # Make sure it's a view of the original data
            assert np.shares_memory(x_val, outputs[0].storage[0])
            # Confirm the broadcasted value in the output
            assert np.array_equiv(outputs[0].storage[0], 2039)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_memory_leak(self, inplace):
        import gc

        n = 100_000

        x = aesara.shared(np.ones(n, dtype=np.float64))

        y = x.dimshuffle([0, "x"])
        y.owner.op.inplace = inplace

        f = aesara.function([], y, mode=Mode(optimizer=None))

        assert len(f.maker.fgraph.apply_nodes) == 2
        assert isinstance(f.maker.fgraph.toposort()[0].op, DimShuffle)

        assert f.maker.fgraph.toposort()[0].op.inplace is inplace

        tracemalloc.start()

        blocks_last = None
        block_diffs = []
        for i in range(50):
            x.set_value(np.ones(n))
            _ = f()
            _ = gc.collect()
            blocks_i, _ = tracemalloc.get_traced_memory()
            if blocks_last is not None:
                blocks_diff = (blocks_i - blocks_last) // 10**3
                block_diffs.append(blocks_diff)
            blocks_last = blocks_i

        tracemalloc.stop()
        assert np.allclose(np.mean(block_diffs), 0)

    def test_static_shape(self):
        x = tensor(np.float64, shape=(1, 2), name="x")
        y = x.dimshuffle([0, 1, "x"])
        assert y.type.shape == (1, 2, 1)


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
        return np.asarray(np.random.random(shp), dtype=aesara.config.floatX)

    def rand_cval(self, shp):
        return np.asarray(np.random.random(shp), dtype=aesara.config.floatX)

    def with_linker(self, linker, op, type, rand_val):
        for shape_info in ("complete", "only_broadcastable", "none"):
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
                if shape_info == "complete":
                    x_type = type(aesara.config.floatX, shape=xsh)
                    y_type = type(aesara.config.floatX, shape=ysh)
                elif shape_info == "only_broadcastable":
                    # This condition is here for backwards compatibility, when the only
                    # type shape provided by Aesara was broadcastable/non-broadcastable
                    x_type = type(
                        aesara.config.floatX,
                        shape=tuple(s if s == 1 else None for s in xsh),
                    )
                    y_type = type(
                        aesara.config.floatX,
                        shape=tuple(s if s == 1 else None for s in ysh),
                    )
                else:
                    x_type = type(aesara.config.floatX, shape=[None for _ in xsh])
                    y_type = type(aesara.config.floatX, shape=[None for _ in ysh])

                x = x_type("x")
                y = y_type("y")
                e = op(aes.add)(x, y)
                f = make_function(copy(linker).accept(FunctionGraph([x, y], [e])))
                xv = rand_val(xsh)
                yv = rand_val(ysh)
                zv = xv + yv

                unittest_tools.assert_allclose(f(xv, yv), zv)

                # test Elemwise.infer_shape
                # the Shape op don't implement c_code!
                if isinstance(linker, PerformLinker):
                    x = x_type("x")
                    y = y_type("y")
                    e = op(aes.add)(x, y)
                    f = make_function(
                        copy(linker).accept(FunctionGraph([x, y], [e.shape]))
                    )
                    assert tuple(f(xv, yv)) == tuple(zv.shape)

    def with_linker_inplace(self, linker, op, type, rand_val):
        for shape_info in ("complete", "only_broadcastable", "none"):
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
                if shape_info == "complete":
                    x_type = type(aesara.config.floatX, shape=xsh)
                    y_type = type(aesara.config.floatX, shape=ysh)
                elif shape_info == "only_broadcastable":
                    # This condition is here for backwards compatibility, when the only
                    # type shape provided by Aesara was broadcastable/non-broadcastable
                    x_type = type(
                        aesara.config.floatX,
                        shape=tuple(s if s == 1 else None for s in xsh),
                    )
                    y_type = type(
                        aesara.config.floatX,
                        shape=tuple(s if s == 1 else None for s in ysh),
                    )
                else:
                    x_type = type(aesara.config.floatX, shape=[None for _ in xsh])
                    y_type = type(aesara.config.floatX, shape=[None for _ in ysh])

                x = x_type("x")
                y = y_type("y")
                e = op(aes.Add(aes.transfer_type(0)), {0: 0})(x, y)
                f = make_function(copy(linker).accept(FunctionGraph([x, y], [e])))
                xv = rand_val(xsh)
                yv = rand_val(ysh)
                zv = xv + yv

                f(xv, yv)

                assert (xv == zv).all()
                # test Elemwise.infer_shape
                # the Shape op don't implement c_code!
                if isinstance(linker, PerformLinker):
                    x = x_type("x")
                    y = y_type("y")
                    e = op(aes.Add(aes.transfer_type(0)), {0: 0})(x, y)
                    f = make_function(
                        copy(linker).accept(FunctionGraph([x, y], [e.shape]))
                    )
                    xv = rand_val(xsh)
                    yv = rand_val(ysh)
                    zv = xv + yv
                    assert xv.shape == zv.shape
                    assert tuple(f(xv, yv)) == zv.shape

    def test_perform(self):
        self.with_linker(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        self.with_linker(CLinker(), self.cop, self.ctype, self.rand_cval)

    def test_perform_inplace(self):
        self.with_linker_inplace(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_inplace(self):
        self.with_linker_inplace(CLinker(), self.cop, self.ctype, self.rand_cval)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_fill(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, shape=(None, None))("x")
            y = t(aesara.config.floatX, shape=(1, 1))("y")
            e = op(aes.Second(aes.transfer_type(0)), {0: 0})(x, y)
            f = make_function(linker().accept(FunctionGraph([x, y], [e])))
            xv = rval((5, 5))
            yv = rval((1, 1))
            f(xv, yv)
            assert (xv == yv).all()

    def test_fill_var(self):
        x = matrix()
        x.fill(3)

    def test_fill_grad(self):
        x = TensorType(config.floatX, shape=(None, 1, None))("x")
        y = TensorType(config.floatX, shape=(None, 1, None))("y")
        e = second(x, y)
        # TODO FIXME: Make this a real test and assert something here!
        aesara.grad(e.sum(), y)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_weird_strides(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, shape=(None,) * 5)("x")
            y = t(aesara.config.floatX, shape=(None,) * 5)("y")
            e = op(aes.add)(x, y)
            f = make_function(linker().accept(FunctionGraph([x, y], [e])))
            xv = rval((2, 2, 2, 2, 2))
            yv = rval((2, 2, 2, 2, 2)).transpose(4, 0, 3, 1, 2)
            zv = xv + yv
            assert (f(xv, yv) == zv).all()

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_same_inputs(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, shape=(None,) * 2)("x")
            e = op(aes.add)(x, x)
            f = make_function(linker().accept(FunctionGraph([x], [e])))
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
        scalar_op=aes.add,
        dtype="floatX",
        pre_scalar_op=None,
        test_nan=False,
        tensor_op=None,
    ):
        for xsh, tosum in self.cases:
            if dtype == "floatX":
                dtype = aesara.config.floatX
            x = self.type(
                dtype, shape=tuple(entry if entry == 1 else None for entry in xsh)
            )("x")
            d = {}
            if pre_scalar_op is not None:
                d = {"pre_scalar_op": pre_scalar_op}
            if tensor_op is None:
                e = as_tensor_variable(self.op(scalar_op, axis=tosum, **d)(x))
            else:
                e = as_tensor_variable(tensor_op(x, axis=tosum, **d))

            if tosum is None:
                tosum = list(range(len(xsh)))

            f = aesara.function([x], e, mode=mode, on_unused_input="ignore")
            xv = np.asarray(np.random.random(xsh))

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

            if len(tosum) > 1 and any(a < 0 for a in tosum):
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
            if tensor_op == at_all:
                for axis in reversed(sorted(tosum)):
                    zv = np.all(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif tensor_op == at_any:
                for axis in reversed(sorted(tosum)):
                    zv = np.any(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif scalar_op == aes.add:
                for axis in reversed(sorted(tosum)):
                    zv = np.add.reduce(zv, axis)
                if dtype == "bool":
                    # np.add of a bool upcast, while CAReduce don't
                    zv = zv.astype(dtype)
            elif scalar_op == aes.mul:
                for axis in reversed(sorted(tosum)):
                    zv = np.multiply.reduce(zv, axis)
            elif scalar_op == aes.scalar_maximum:
                # There is no identity value for the maximum function
                # So we can't support shape of dimensions 0.
                if np.prod(zv.shape) == 0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = np.maximum.reduce(zv, axis)
            elif scalar_op == aes.scalar_minimum:
                # There is no identity value for the minimum function
                # So we can't support shape of dimensions 0.
                if np.prod(zv.shape) == 0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = np.minimum.reduce(zv, axis)
            elif scalar_op == aes.or_:
                for axis in reversed(sorted(tosum)):
                    zv = np.bitwise_or.reduce(zv, axis)
            elif scalar_op == aes.and_:
                for axis in reversed(sorted(tosum)):
                    zv = reduce_bitwise_and(zv, axis, dtype=dtype)
            elif scalar_op == aes.xor:
                # There is no identity value for the xor function
                # So we can't support shape of dimensions 0.
                if np.prod(zv.shape) == 0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = np.bitwise_xor.reduce(zv, axis)
            else:
                raise NotImplementedError(
                    f"Test for CAReduce with scalar_op {scalar_op} not implemented"
                )

            if test_nan:
                assert self.type.values_eq(f(xv), zv), (f(xv), zv)
            else:
                f_xv = f(xv)
                assert f_xv.shape == zv.shape, (f_xv, zv)
                utt.assert_allclose(zv, f_xv)

            x = self.type(
                dtype, shape=tuple(entry if entry == 1 else None for entry in xsh)
            )("x")
            if tensor_op is None:
                e = self.op(scalar_op, axis=tosum)(x)
            else:
                e = tensor_op(x, axis=tosum)
            if tosum is None:
                tosum = list(range(len(xsh)))
            f = aesara.function([x], e.shape, mode=mode, on_unused_input="ignore")
            if not (
                scalar_op in [aes.scalar_maximum, aes.scalar_minimum]
                and (xsh == () or np.prod(xsh) == 0)
            ):
                assert all(f(xv) == zv.shape)

    def test_perform_noopt(self):
        self.with_mode(Mode(linker="py", optimizer=None), aes.add, dtype="floatX")

    def test_perform(self):
        for dtype in ["bool", "floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_mode(Mode(linker="py"), aes.add, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.mul, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.scalar_maximum, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.scalar_minimum, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.and_, dtype=dtype, tensor_op=at_all)
            self.with_mode(Mode(linker="py"), aes.or_, dtype=dtype, tensor_op=at_any)
        for dtype in ["int8", "uint8"]:
            self.with_mode(Mode(linker="py"), aes.or_, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.and_, dtype=dtype)
            self.with_mode(Mode(linker="py"), aes.xor, dtype=dtype)

    def test_perform_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_mode(Mode(linker="py"), aes.add, dtype=dtype, test_nan=True)
            self.with_mode(Mode(linker="py"), aes.mul, dtype=dtype, test_nan=True)
            self.with_mode(
                Mode(linker="py"), aes.scalar_maximum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="py"), aes.scalar_minimum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="py"),
                aes.or_,
                dtype=dtype,
                test_nan=True,
                tensor_op=at_any,
            )
            self.with_mode(
                Mode(linker="py"),
                aes.and_,
                dtype=dtype,
                test_nan=True,
                tensor_op=at_all,
            )

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_noopt(self):
        # We need to make sure that we cover the corner cases that
        # optimizations normally cover
        self.with_mode(Mode(linker="c", optimizer=None), aes.add, dtype="floatX")

    @pytest.mark.slow
    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        for dtype in ["bool", "floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), aes.add, dtype=dtype)
            self.with_mode(Mode(linker="c"), aes.mul, dtype=dtype)
        for dtype in ["bool", "floatX", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), aes.scalar_minimum, dtype=dtype)
            self.with_mode(Mode(linker="c"), aes.scalar_maximum, dtype=dtype)
            self.with_mode(Mode(linker="c"), aes.and_, dtype=dtype, tensor_op=at_all)
            self.with_mode(Mode(linker="c"), aes.or_, dtype=dtype, tensor_op=at_any)
        for dtype in ["bool", "int8", "uint8"]:
            self.with_mode(Mode(linker="c"), aes.or_, dtype=dtype)
            self.with_mode(Mode(linker="c"), aes.and_, dtype=dtype)
            self.with_mode(Mode(linker="c"), aes.xor, dtype=dtype)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_mode(Mode(linker="c"), aes.add, dtype=dtype, test_nan=True)
            self.with_mode(Mode(linker="c"), aes.mul, dtype=dtype, test_nan=True)
        for dtype in ["floatX"]:
            self.with_mode(
                Mode(linker="c"), aes.scalar_minimum, dtype=dtype, test_nan=True
            )
            self.with_mode(
                Mode(linker="c"), aes.scalar_maximum, dtype=dtype, test_nan=True
            )

    def test_infer_shape(self, dtype=None, pre_scalar_op=None):
        if dtype is None:
            dtype = aesara.config.floatX
        for xsh, tosum in self.cases:
            x = self.type(
                dtype, shape=tuple(entry if entry == 1 else None for entry in xsh)
            )("x")
            if pre_scalar_op is not None:
                x = pre_scalar_op(x)
            if tosum is None:
                tosum = list(range(len(xsh)))
            xv = np.asarray(np.random.random(xsh), dtype=dtype)
            d = {}
            if pre_scalar_op is not None:
                xv = x.eval({x.owner.inputs[0]: xv})
                d = {pre_scalar_op: pre_scalar_op}
            self._compile_and_check(
                [x],
                [self.op(aes.add, axis=tosum, *d)(x)],
                [xv],
                self.op,
                ["local_cut_useless_reduce"],
                warn=0 not in xsh,
            )

    def test_str(self):
        op = CAReduce(aes.add, axis=None)
        assert str(op) == "CAReduce{add}"
        op = CAReduce(aes.add, axis=(1,))
        assert str(op) == "CAReduce{add}{axis=[1]}"

        op = CAReduce(aes.add, axis=None, acc_dtype="float64")
        assert str(op) == "CAReduce{add}{acc_dtype=float64}"
        op = CAReduce(aes.add, axis=(1,), acc_dtype="float64")
        assert str(op) == "CAReduce{add}{axis=[1], acc_dtype=float64}"

    def test_repeated_axis(self):
        x = vector("x")
        with pytest.raises(ValueError, match="repeated axis"):
            self.op(aes.add, axis=(0, 0))(x)

    def test_scalar_input(self):
        x = scalar("x")

        assert self.op(aes.add, axis=(-1,))(x).eval({x: 5}) == 5

        with pytest.raises(
            np.AxisError,
            match=re.escape("axis (-2,) is out of bounds for array of dimension 0"),
        ):
            self.op(aes.add, axis=(-2,))(x)


class TestBitOpReduceGrad:
    def setup_method(self):
        self.rng = np.random.default_rng(unittest_tools.fetch_seed())

    def test_all_grad(self):
        x = bmatrix("x")
        x_all = x.all()
        gx = aesara.grad(x_all, x)
        f = aesara.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype("int8")
        for x_val in (x_random, np.zeros_like(x_random), np.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert np.all(gx_val == 0)

    def test_any_grad(self):
        x = bmatrix("x")
        x_all = x.any()
        gx = aesara.grad(x_all, x)
        f = aesara.function([x], gx)
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
        dx, dy = aesara.grad(z, [x, y])

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
            dtype = aesara.config.floatX
            t_left = TensorType(
                dtype, shape=tuple(entry if entry == 1 else None for entry in s_left)
            )()
            t_right = TensorType(
                dtype, shape=tuple(entry if entry == 1 else None for entry in s_right)
            )()
            t_left_val = np.zeros(s_left, dtype=dtype)
            t_right_val = np.zeros(s_right, dtype=dtype)
            self._compile_and_check(
                [t_left, t_right],
                [Elemwise(aes.add)(t_left, t_right)],
                [t_left_val, t_right_val],
                Elemwise,
            )

    def test_input_dimensions_overflow(self):
        # Elemwise.perform used to compute the product
        # of input shapes to check if there was a zero in them,
        # it overflowed in this case.
        a, b, c, d, e, f = vectors("abcdef")
        s = a + b + c + d + e + f
        g = aesara.function([a, b, c, d, e, f], s, mode=Mode(linker="py"))
        g(*[np.zeros(2**11, config.floatX) for i in range(6)])

    def check_input_dimensions_match(self, mode):
        """Make sure that our input validation works correctly and doesn't
        throw erroneous broadcast-based errors.
        """
        x_v = matrix("x")
        m_v = vector("m")

        x = np.array([[-1.32720483], [0.23442016]]).astype(config.floatX)
        m = np.array([0.0, 0.0]).astype(config.floatX)

        z_v = x_v - m_v
        f = aesara.function([x_v, m_v], z_v, mode=mode)

        res = f(x, m)

        assert np.array_equal(res, x - m)

    def test_input_dimensions_match_python(self):
        self.check_input_dimensions_match(Mode(linker="py"))

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_input_dimensions_match_c(self):
        self.check_input_dimensions_match(Mode(linker="c"))

    def test_str(self):
        op = Elemwise(aes.add, inplace_pattern=None, name=None)
        assert str(op) == "Elemwise{add}"
        op = Elemwise(aes.add, inplace_pattern={0: 0}, name=None)
        assert str(op) == "Elemwise{add}[(0, 0)]"
        op = Elemwise(aes.add, inplace_pattern=None, name="my_op")
        assert str(op) == "my_op"

    def test_partial_static_shape_info(self):
        """Make sure that `Elemwise.infer_shape` can handle changes in the static shape information during rewriting."""

        x = TensorType("floatX", shape=(None, None))()
        z = Elemwise(aes.add)(x, x)

        x_inferred_shape = (aes.constant(1), aes.constant(1))

        res_shape = z.owner.op.infer_shape(
            None, z.owner, [x_inferred_shape, x_inferred_shape]
        )

        assert len(res_shape) == 1
        assert len(res_shape[0]) == 2
        assert aesara.get_scalar_constant_value(res_shape[0][0]) == 1
        assert aesara.get_scalar_constant_value(res_shape[0][1]) == 1

    def test_multi_output(self):
        class CustomElemwise(Elemwise):
            def make_node(self, *args):
                res = super().make_node(*args)
                return Apply(
                    self,
                    res.inputs,
                    # Return two outputs
                    [
                        TensorType(dtype="float64", shape=(None, None))()
                        for i in range(2)
                    ],
                )

        z_1, z_2 = CustomElemwise(aes.add)(
            as_tensor_variable(np.eye(1)), as_tensor_variable(np.eye(1))
        )

        in_1_shape = (aes.constant(1), aes.constant(1))

        with pytest.raises(ShapeError):
            z_1.owner.op.infer_shape(None, z_1.owner, [in_1_shape, in_1_shape])

    def test_shape_types(self):
        x = tensor(np.float64, (None, 1))
        y = tensor(np.float64, (50, 10))

        z = x * y

        assert isinstance(z.owner.op, Elemwise)

        (out_shape,) = z.owner.op.infer_shape(None, z.owner, [(lscalar(), 1), (50, 10)])

        assert all(isinstance(v.type, TensorType) for v in out_shape)

    def test_static_shape_unary(self):
        x = tensor("float64", shape=(None, 0, 1, 5))
        assert exp(x).type.shape == (None, 0, 1, 5)

    def test_static_shape_binary(self):
        x = tensor("float64", shape=(None, 5))
        y = tensor("float64", shape=(None, 5))
        assert (x + y).type.shape == (None, 5)

        x = tensor("float64", shape=(None, 5))
        y = tensor("float64", shape=(10, 5))
        assert (x + y).type.shape == (10, 5)

        x = tensor("float64", shape=(1, 5))
        y = tensor("float64", shape=(10, 5))
        assert (x + y).type.shape == (10, 5)

        x = tensor("float64", shape=(None, 1))
        y = tensor("float64", shape=(1, 1))
        assert (x + y).type.shape == (None, 1)

        x = tensor("float64", shape=(0, 0, 0))
        y = tensor("float64", shape=(0, 1, None))
        assert (x + y).type.shape == (0, 0, 0)

    def test_invalid_static_shape(self):
        x = tensor("float64", shape=(2,))
        y = tensor("float64", shape=(3,))
        with pytest.raises(
            ValueError,
            match=re.escape("Incompatible Elemwise input shapes [(2,), (3,)]"),
        ):
            x + y


def test_not_implemented_elemwise_grad():
    # Regression test for unimplemented gradient in an Elemwise Op.

    class TestOp(aes.ScalarOp):
        def __init__(self):
            self.output_types_preference = aes.upgrade_to_float

        def impl(self, n, x):
            return x * n

        def grad(self, inputs, gout):
            (n, x) = inputs
            (gz,) = gout
            dy_dx = n
            return [aesara.gradient.grad_not_implemented(self, 0, n), gz * dy_dx]

    test_op = Elemwise(TestOp())
    x = scalar()
    assert isinstance(aesara.gradient.grad(test_op(2, x), x), Variable)

    # Verify that trying to use the not implemented gradient fails.
    with pytest.raises(aesara.gradient.NullTypeGradError):
        aesara.gradient.grad(test_op(x, 2), x)

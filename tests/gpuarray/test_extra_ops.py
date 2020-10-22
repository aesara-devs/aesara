from functools import partial
from itertools import product

import numpy as np
import pytest

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from tests.gpuarray.config import mode_with_gpu, test_ctx_name
from tests.tensor.test_extra_ops import TestCumOp
from theano.gpuarray.extra_ops import GpuCumOp
from theano.gpuarray.type import get_context
from theano.tensor.extra_ops import CumOp


class TestGpuCumOp(TestCumOp):
    mode = mode_with_gpu

    def setup_method(self):
        super().setup_method()
        test_ctx = get_context(test_ctx_name)
        if test_ctx.kind != b"cuda":
            pytest.skip("Cuda specific tests")
        self.max_threads_dim0 = test_ctx.maxlsize0
        self.max_grid_size1 = test_ctx.maxgsize2
        self.op_class = CumOp

        # The CPU implementation is not so accurate, which throws out DebugMode.
        # Since propagating .tag.values_eq_approx to the output of every
        # GpuFromHost seems overkill, we just relax the rtol for these tests
        self.old_rtol = tt.float32_rtol
        tt.float32_rtol *= 2

    def teardown_method(self):
        super().teardown_method()
        # Restore rtol
        tt.float32_rtol = self.old_rtol

    @pytest.mark.skipif(
        theano.config.floatX != "float32",
        reason="Gpucumop not implemented for dtype %s" % theano.config.floatX,
    )
    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_infer_shape(self, mode):
        op_class = partial(self.op_class, mode=mode)
        x = tt.tensor3("x")
        a = np.random.random((3, 5, 2)).astype(theano.config.floatX)

        for axis in range(-len(a.shape), len(a.shape)):
            self._compile_and_check([x], [op_class(axis=axis)(x)], [a], GpuCumOp)

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_Strides1D(self, mode):
        op_class = partial(self.op_class, mode=mode)
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        x = tt.fvector("x")

        for axis in [0, None, -1]:
            a = np.random.random((42,)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode
            )

            slicings = [
                slice(None, None, None),  # Normal strides
                slice(None, None, 2),  # Stepped strides
                slice(None, None, -1),  # Negative strides
            ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function(
                    [x], op_class(axis=axis)(x[slicing]), mode=self.mode
                )
                assert [
                    n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)
                ]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(
                    np_func(a[slicing], axis=axis), cumop_function(a[slicing])
                )

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_Strides2D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        x = tt.fmatrix("x")

        for axis in [0, 1, None, -1, -2]:
            a = np.random.random((42, 30)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode
            )

            slicings = [
                slice(None, None, None),  # Normal strides
                slice(None, None, 2),  # Stepped strides
                slice(None, None, -1),  # Negative strides
            ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function(
                    [x], op_class(axis=axis)(x[slicing]), mode=self.mode
                )
                assert [
                    n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)
                ]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(
                    np_func(a[slicing], axis=axis), cumop_function(a[slicing])
                )

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_Strides3D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        x = tt.ftensor3("x")

        for axis in [0, 1, 2, None, -1, -2, -3]:
            a = np.random.random((42, 30, 25)).astype("float32")
            cumop_function = theano.function(
                [x], op_class(axis=axis)(x), mode=self.mode
            )

            slicings = [
                slice(None, None, None),  # Normal strides
                slice(None, None, 2),  # Stepped strides
                slice(None, None, -1),  # Negative strides
            ]

            # Cartesian product of all slicings to test.
            for slicing in product(slicings, repeat=x.ndim):
                f = theano.function(
                    [x], op_class(axis=axis)(x[slicing]), mode=self.mode
                )
                assert [
                    n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)
                ]
                utt.assert_allclose(np_func(a[slicing], axis=axis), f(a))
                utt.assert_allclose(
                    np_func(a[slicing], axis=axis), cumop_function(a[slicing])
                )

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_GpuCumOp1D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = tt.fvector("x")
        f = theano.function([x], op_class(axis=0)(x), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)]

        # Extensive testing for the first 1025 sizes
        a = np.random.random(1025).astype("float32")
        for i in range(a.shape[0]):
            utt.assert_allclose(np_func(a[:i]), f(a[:i]))

        # Use multiple GPU threadblocks
        a = np.random.random((block_max_size + 2,)).astype("float32")
        utt.assert_allclose(np_func(a), f(a))

        # Use recursive cumop
        a = np.ones((block_max_size * (block_max_size + 1) + 2,), dtype="float32")
        utt.assert_allclose(np_func(a), f(a))

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_GpuCumOp2D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = tt.fmatrix("x")
        for shape_axis, axis in zip([0, 1, 0, 1, 0], [0, 1, None, -1, -2]):
            f = theano.function([x], op_class(axis=axis)(x), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.random(a_shape).astype("float32")
            slices = [slice(None), slice(None)]
            for i in range(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np_func(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks
            a_shape = [5, 5]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use multiple GPU gridblocks
            a_shape = [4, 4]
            a_shape[1 - shape_axis] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a), rtol=5e-5)

            # Use recursive cumop
            a_shape = [3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np_func(a, axis=axis), f(a))

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_GpuCumOp3D(self, mode):
        np_func = dict(add=np.cumsum, mul=np.cumprod)[mode]
        op_class = partial(self.op_class, mode=mode)
        block_max_size = self.max_threads_dim0 * 2

        x = tt.ftensor3("x")
        for shape_axis, axis in zip([0, 1, 2, 0, 2, 1, 0], [0, 1, 2, None, -1, -2, -3]):
            f = theano.function([x], op_class(axis=axis)(x), mode=self.mode)
            assert [n for n in f.maker.fgraph.toposort() if isinstance(n.op, GpuCumOp)]

            # Extensive testing for the first 1025 sizes
            a_shape = [5, 5, 5]
            a_shape[shape_axis] = 1025
            a = np.random.rand(*a_shape).astype("float32")
            slices = [slice(None), slice(None), slice(None)]
            for i in range(a.shape[shape_axis]):
                slices[shape_axis] = slice(i)
                fa = f(a[slices])
                npa = np_func(a[slices], axis=axis)
                utt.assert_allclose(npa, fa)

            # Use multiple GPU threadblocks (along accumulation axis)
            a_shape = [2, 2, 2]
            a_shape[shape_axis] = block_max_size + 2
            a = np.random.random(a_shape).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use multiple GPU gridblocks (not along accumulation axis)
            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 1) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            a_shape = [5, 5, 5]
            a_shape[(shape_axis + 2) % 3] = self.max_grid_size1 + 1
            a = np.random.random(a_shape).astype("float32")
            if axis is None:
                # Avoid floating point error
                a = np.sign(a - 0.5).astype("float32")
            utt.assert_allclose(np_func(a, axis=axis), f(a))

            # Use recursive cumop (along accumulation axis)
            a_shape = [3, 3, 3]
            a_shape[shape_axis] = block_max_size * (block_max_size + 1) + 2
            a = np.random.random(a_shape).astype("float32")
            a = np.sign(a - 0.5).astype("float32")  # Avoid floating point error
            utt.assert_allclose(np_func(a, axis=axis), f(a))

    @pytest.mark.parametrized("mode", ["mul", "add"])
    def test_GpuCumOp4D(self, mode):
        op_class = partial(self.op_class, mode=mode)
        # Should not use the GPU version.
        x = tt.ftensor4("x")
        f = theano.function([x], op_class(axis=1)(x), mode=self.mode)
        assert [n for n in f.maker.fgraph.toposort() if isinstance(n.op, CumOp)]

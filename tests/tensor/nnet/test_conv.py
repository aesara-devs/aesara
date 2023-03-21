import time

import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.compile.mode import Mode
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import _allclose, exp
from aesara.tensor.nnet import conv, conv2d
from aesara.tensor.type import dmatrix, dtensor3, dtensor4, dvector, scalar, tensor4
from tests import unittest_tools as utt


@pytest.mark.skipif(
    aesara.config.cxx == "",
    reason="conv2d tests need SciPy or a c++ compiler",
)
class TestConv2D(utt.InferShapeTester):
    # This class contains tests for the legacy 2d convolution,
    # but will also be inherited from for other implementations
    mode = None
    dtype = aesara.config.floatX
    # This will be set to the appropriate function in the inherited classes.
    # The call to `staticmethod` is necessary to prevent Python from passing
    # `self` as the first argument.
    conv2d = staticmethod(conv2d)

    def setup_method(self):
        self.input = tensor4("input", dtype=self.dtype)
        self.input.name = "default_V"
        self.filters = tensor4("filters", dtype=self.dtype)
        self.filters.name = "default_filters"
        super().setup_method()

    def validate(
        self,
        image_shape,
        filter_shape,
        border_mode="valid",
        subsample=(1, 1),
        N_image_shape=None,
        N_filter_shape=None,
        input=None,
        filters=None,
        unroll_batch=None,
        unroll_kern=None,
        unroll_patch=None,
        verify_grad=True,
        should_raise=False,
    ):
        """
        :param image_shape: The constant shape info passed to conv2d.
        :param filter_shape: The constant shape info passed to conv2d.

        :param N_image_shape: None(default to image_shape) or tuple of
                              4 elements with the shape of the input image

        :param N_filter_shape: None(default to filter_shape) or tuple
                               of 4 elements with the shape of the
                               input filter

        """
        if N_image_shape is None:
            N_image_shape = [
                at.get_scalar_constant_value(at.as_tensor_variable(x))
                for x in image_shape
            ]
        if N_filter_shape is None:
            N_filter_shape = [
                at.get_scalar_constant_value(at.as_tensor_variable(x))
                for x in filter_shape
            ]

        if input is None:
            input = self.input
        if not filters:
            filters = self.filters

        # AESARA IMPLEMENTATION

        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            # define aesara graph and function
            input.name = "input"
            filters.name = "filters"
            with pytest.warns(DeprecationWarning):
                rval = conv.conv2d(
                    input,
                    filters,
                    image_shape,
                    filter_shape,
                    border_mode,
                    subsample,
                    unroll_batch=unroll_batch,
                    unroll_kern=unroll_kern,
                    unroll_patch=unroll_patch,
                )
            rval.name = "conv_output"
            return rval

        output = sym_conv2d(input, filters)
        output.name = f"conv2d({input.name},{filters.name})"
        aesara_conv = aesara.function([input, filters], output, mode=self.mode)

        # initialize input and compute result
        image_data = np.random.random(N_image_shape).astype(self.dtype)
        filter_data = np.random.random(N_filter_shape).astype(self.dtype)
        try:
            aesara_output = aesara_conv(image_data, filter_data)
        except ValueError:
            if not should_raise:
                raise
            return
        else:
            if should_raise:
                raise Exception("ConvOp should have generated an error")

        # REFERENCE IMPLEMENTATION
        s = 1.0
        orig_image_data = image_data
        if border_mode != "full":
            s = -1.0
        out_shape2d = (
            np.array(N_image_shape[-2:]) + s * np.array(N_filter_shape[-2:]) - s
        )
        out_shape2d = np.ceil(out_shape2d / np.array(subsample))
        # avoid numpy deprecation
        out_shape2d = out_shape2d.astype("int32")
        out_shape = (N_image_shape[0], N_filter_shape[0]) + tuple(out_shape2d)
        ref_output = np.zeros(out_shape)

        # loop over output feature maps
        ref_output.fill(0)
        if border_mode == "full":
            image_data2 = np.zeros(
                (
                    N_image_shape[0],
                    N_image_shape[1],
                    N_image_shape[2] + 2 * N_filter_shape[2] - 2,
                    N_image_shape[3] + 2 * N_filter_shape[3] - 2,
                )
            )
            image_data2[
                :,
                :,
                N_filter_shape[2] - 1 : N_filter_shape[2] - 1 + N_image_shape[2],
                N_filter_shape[3] - 1 : N_filter_shape[3] - 1 + N_image_shape[3],
            ] = image_data
            image_data = image_data2
            N_image_shape = image_data.shape
        for bb in range(N_image_shape[0]):
            for nn in range(N_filter_shape[0]):
                for im0 in range(N_image_shape[1]):
                    filter2d = filter_data[nn, im0, :, :]
                    image2d = image_data[bb, im0, :, :]
                    for row in range(ref_output.shape[2]):
                        irow = row * subsample[0]  # image row
                        for col in range(ref_output.shape[3]):
                            icol = col * subsample[1]  # image col
                            ref_output[bb, nn, row, col] += (
                                image2d[
                                    irow : irow + N_filter_shape[2],
                                    icol : icol + N_filter_shape[3],
                                ]
                                * filter2d[::-1, ::-1]
                            ).sum()

        assert _allclose(aesara_output, ref_output)

        # TEST GRADIENT
        if verify_grad:
            utt.verify_grad(sym_conv2d, [orig_image_data, filter_data])

    def test_basic1(self):
        # Tests that basic convolutions work for odd and even
        # dimensions of image and filter shapes, as well as rectangular
        # images and filters.

        self.validate((2, 2, 3, 3), (2, 2, 2, 2), "valid", verify_grad=False)

    def test_basic(self):
        # Tests that basic convolutions work for odd and even
        # dimensions of image and filter shapes, as well as rectangular
        # images and filters.

        self.validate((3, 2, 8, 8), (4, 2, 5, 5), "valid", verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "valid")
        self.validate((3, 2, 7, 5), (5, 2, 3, 2), "valid", verify_grad=False)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), "full", verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "full")
        # test filter same size as input

    def test_uint_image_shape_datatype(self):
        # Tests for uint datatype in image_shape.

        self.validate((2, 2, 3, np.uint8(3)), (3, 2, 3, 3), "valid", verify_grad=False)
        self.validate((np.uint16(2), 2, 3, 3), (3, 2, 3, 3), "valid", verify_grad=False)
        self.validate((2, np.uint32(2), 3, 3), (3, 2, 3, 3), "valid", verify_grad=False)

    def test_uint_filter_shape_datatype(self):
        # Tests for uint datatype in filter_shape

        self.validate((3, 2, 3, 3), (2, 2, 3, np.uint8(3)), "valid", verify_grad=False)
        self.validate((3, 2, 3, 3), (np.uint16(2), 2, 3, 3), "valid", verify_grad=False)
        self.validate((3, 2, 3, 3), (2, np.uint32(2), 3, 3), "valid", verify_grad=False)

    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), "full")
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), "valid")

    def test_unroll_patch_true(self):
        # Test basic convs with True.

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "valid", unroll_patch=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "full", unroll_patch=True)
        self.validate(
            (3, 2, 3, 3), (4, 2, 3, 3), "valid", unroll_patch=True, verify_grad=False
        )

    def test_unroll_patch_false(self):
        # Test basic convs with False.

        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "valid", unroll_patch=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "full", unroll_patch=False)
        self.validate(
            (3, 2, 3, 3), (4, 2, 3, 3), "valid", unroll_patch=False, verify_grad=False
        )

    def test_unroll_patch_true_fail(self):
        # Test basic convs with True.

        self.validate(
            (3, 2, 7, 5),
            (5, 2, 2, 3),
            "valid",
            unroll_patch=True,
            N_image_shape=(1, 3, 3, 3),
            N_filter_shape=(6, 3, 2, 2),
            should_raise=True,
        )
        self.validate(
            (3, 2, 7, 5),
            (5, 2, 2, 3),
            "full",
            unroll_patch=True,
            N_image_shape=(1, 3, 3, 3),
            N_filter_shape=(6, 3, 2, 2),
            should_raise=True,
        )
        self.validate(
            (3, 2, 3, 3),
            (4, 2, 3, 3),
            "valid",
            unroll_patch=True,
            N_image_shape=(1, 3, 3, 3),
            N_filter_shape=(6, 3, 2, 2),
            should_raise=True,
        )

    def test_unroll_special(self):
        # (unroll_kern, unroll_batch) in (0,1),(1,0) is special case.

        self.validate((6, 2, 3, 3), (3, 2, 2, 2), "valid", unroll_batch=1)

    def test_unroll_batch(self):
        # Test mini-batch unrolling for various legal values.

        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate(
            (6, 2, 3, 3), (3, 2, 2, 2), "valid", unroll_batch=2, verify_grad=False
        )
        self.validate(
            (6, 2, 3, 3), (3, 2, 2, 2), "valid", unroll_batch=3, verify_grad=False
        )

    def test_unroll_kern(self):
        # Test kernel unrolling for various legal values.

        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate(
            (2, 3, 3, 3), (6, 3, 2, 2), "valid", unroll_kern=2, verify_grad=False
        )
        self.validate(
            (2, 3, 3, 3), (6, 3, 2, 2), "valid", unroll_kern=3, verify_grad=False
        )

    def test_unroll_batch_kern(self):
        # Test mini-batch unrolling with kernel unrolling for various
        # legal values.

        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate(
            (6, 2, 3, 3),
            (3, 2, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=3,
            verify_grad=False,
        )
        self.validate(
            (6, 2, 3, 3),
            (3, 2, 2, 2),
            "valid",
            unroll_batch=3,
            unroll_kern=3,
            verify_grad=False,
        )
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate(
            (2, 3, 3, 3),
            (6, 3, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=2,
            verify_grad=False,
        )
        self.validate(
            (2, 3, 3, 3),
            (6, 3, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=3,
            verify_grad=False,
        )

    def test_unroll_batch_kern_fail(self):
        # Test mini-batch unrolling with kernel unrolling for various
        # legal values, but pass bad input.  All those test must
        # generate errors

        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate(
            (6, 2, 3, 3),
            (3, 2, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=3,
            N_image_shape=(7, 2, 3, 3),
            N_filter_shape=(3, 2, 2, 2),
            should_raise=True,
        )
        self.validate(
            (6, 2, 3, 3),
            (3, 2, 2, 2),
            "valid",
            unroll_batch=3,
            unroll_kern=3,
            N_image_shape=(6, 2, 3, 3),
            N_filter_shape=(4, 2, 2, 2),
            should_raise=True,
        )
        self.validate(
            (2, 3, 3, 3),
            (6, 3, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=2,
            N_image_shape=(1, 3, 3, 3),
            N_filter_shape=(6, 3, 2, 2),
            should_raise=True,
        )
        self.validate(
            (2, 3, 3, 3),
            (6, 3, 2, 2),
            "valid",
            unroll_batch=2,
            unroll_kern=3,
            N_image_shape=(2, 3, 3, 3),
            N_filter_shape=(5, 3, 2, 2),
            should_raise=True,
        )

    def test_subsample(self):
        # Tests convolution where subsampling != (1,1)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), "full", subsample=(2, 2))

        # Fails as of 2012-07-11
        with pytest.raises(NotImplementedError):
            self.validate((1, 1, 6, 6), (1, 1, 3, 3), "full", subsample=(3, 3))

        # Fails as of 2017-08-10
        with pytest.raises(NotImplementedError):
            self.validate((3, 2, 7, 5), (5, 2, 2, 3), "valid", subsample=(2, 2))
        with pytest.raises(NotImplementedError):
            self.validate((3, 2, 7, 5), (5, 2, 2, 3), "valid", subsample=(2, 1))
        with pytest.raises(NotImplementedError):
            self.validate((1, 1, 6, 6), (1, 1, 3, 3), "valid", subsample=(3, 3))

    def test_shape_Constant_tensor(self):
        # Tests convolution where the {image,filter}_shape is a Constant tensor.

        as_t = at.as_tensor_variable
        self.validate((as_t(3), as_t(2), as_t(7), as_t(5)), (5, 2, 2, 3), "valid")
        self.validate(as_t([3, 2, 7, 5]), (5, 2, 2, 3), "valid")
        self.validate(as_t((3, 2, 7, 5)), (5, 2, 2, 3), "valid")
        self.validate((3, 2, 7, 5), (as_t(5), as_t(2), as_t(2), as_t(3)), "valid")
        self.validate((3, 2, 7, 5), as_t([5, 2, 2, 3]), "valid")
        self.validate((3, 2, 7, 5), as_t((5, 2, 2, 3)), "valid")
        self.validate(as_t([3, 2, 7, 5]), as_t([5, 2, 2, 3]), "full")

    def test_invalid_filter_shape(self):
        # Tests scenario where filter_shape[1] != input_shape[1]

        with pytest.raises(AssertionError):
            self.validate((3, 2, 8, 8), (4, 3, 5, 5), "valid")

    def test_invalid_input_shape(self):
        # Tests that when the shape given at build time is not the same as
        # run time we raise an error

        for unroll_batch in [None, 1, 3]:
            for unroll_kern in [None, 2, 4]:
                for unroll_patch in [None, True, False]:
                    for mode in ["valid", "full"]:
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_image_shape=(2, 2, 8, 8),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_image_shape=(3, 1, 8, 8),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_image_shape=(3, 2, 7, 8),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_image_shape=(3, 2, 8, 7),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )

                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_filter_shape=(3, 2, 5, 5),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_filter_shape=(4, 1, 5, 5),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_filter_shape=(4, 2, 6, 5),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )
                        with pytest.raises(ValueError):
                            self.validate(
                                (3, 2, 8, 8),
                                (4, 2, 5, 5),
                                mode,
                                N_filter_shape=(4, 2, 5, 6),
                                unroll_batch=unroll_batch,
                                unroll_kern=unroll_kern,
                                unroll_patch=unroll_patch,
                            )

    def test_missing_info(self):
        # Test convolutions for various pieces of missing info.

        self.validate(
            None, None, N_image_shape=(3, 2, 8, 8), N_filter_shape=(4, 2, 5, 5)
        )
        self.validate(
            (3, 2, None, None),
            None,
            N_image_shape=(3, 2, 8, 8),
            N_filter_shape=(4, 2, 5, 5),
        )
        self.validate(
            (None, 2, None, None),
            (None, 2, 5, 5),
            N_image_shape=(3, 2, 8, 8),
            N_filter_shape=(4, 2, 5, 5),
        )
        self.validate(
            (3, 2, 8, 8),
            (4, 2, None, 5),
            N_image_shape=(3, 2, 8, 8),
            N_filter_shape=(4, 2, 5, 5),
        )
        self.validate(
            (3, 2, 8, 8),
            (4, 2, 5, None),
            N_image_shape=(3, 2, 8, 8),
            N_filter_shape=(4, 2, 5, 5),
        )

    def test_wrong_info(self):
        # Test convolutions when we don't give a constant as shape information

        i = aesara.scalar.basic.int32()
        with pytest.raises(NotScalarConstantError):
            self.validate(
                (3, 2, 8, i),
                (4, 2, 5, 5),
                N_image_shape=(3, 2, 8, 8),
                N_filter_shape=(4, 2, 5, 5),
            )
        with pytest.raises(NotScalarConstantError):
            self.validate(
                (3, 2, 8, 8),
                (4, 2, 5, i),
                N_image_shape=(3, 2, 8, 8),
                N_filter_shape=(4, 2, 5, 5),
            )

    def test_full_mode(self):
        # Tests basic convolution in full mode and case where filter
        # is larger than the input image.

        self.validate((3, 2, 5, 5), (4, 2, 8, 8), "full")

        def f():
            self.validate((3, 2, 5, 5), (4, 2, 8, 8), "valid")

        with pytest.raises(Exception):
            f()

    def test_wrong_input(self):
        # Make sure errors are raised when image and kernel are not 4D tensors

        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8), (4, 2, 5, 5), "valid", input=dmatrix())
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8), (4, 2, 5, 5), "valid", filters=dvector())
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8), (4, 2, 5, 5), "valid", input=dtensor3())

    def test_gcc_crash(self):
        # gcc 4.3.0 20080428 (Red Hat 4.3.0-8)
        #
        # crashed in this following case. I changed the c code to don't hit
        # gcc bug. So it should not crash anymore

        self.validate((1, 10, 213, 129), (46, 10, 212, 1), "valid", verify_grad=False)

    def speed(self):
        n_calls = 20000
        print("n_calls", n_calls)
        for border_mode in ["valid", "full"]:
            print()
            print(border_mode)
            for openmp in [False, True]:
                print("OpenMP", openmp)
                image_shapes = [
                    (1, 5, 6, 6),
                    (10, 5, 6, 6)
                    # (10, 10, 16, 16),
                    # (10, 10, 32, 32)]
                ]
                print("image_shape", image_shapes)
                for image_shape in image_shapes:
                    filter_shapes = [(1, 5, 4, 4), (2, 5, 4, 4), (5, 5, 4, 4)]
                    print("filter_shapes", filter_shapes)
                    for filter_shape in filter_shapes:
                        input = aesara.shared(np.random.random(image_shape))
                        filters = aesara.shared(np.random.random(filter_shape))

                        with pytest.warns(DeprecationWarning):
                            output = conv.conv2d(
                                input,
                                filters,
                                image_shape,
                                filter_shape,
                                border_mode,
                                unroll_patch=True,
                                openmp=openmp,
                            )
                        mode = Mode(
                            linker=aesara.link.vm.VMLinker(
                                allow_gc=False, use_cloop=True
                            )
                        )
                        aesara_conv = aesara.function([], output, mode=mode)
                        t1 = time.perf_counter()
                        aesara_conv.vm(n_calls=n_calls)
                        t2 = time.perf_counter()
                        print(t2 - t1, end=" ")
                    print()

    def test_infer_shape(self):
        # Note: infer_shape is incomplete and thus input and filter shapes
        # must be provided explicitly

        rng = np.random.default_rng(280284)

        def rand(*shape):
            r = np.asarray(rng.random(shape), dtype="float64")
            return r * 2 - 1

        adtens = dtensor4()
        bdtens = dtensor4()
        aivec_val = [4, 5, 6, 3]
        bivec_val = [7, 5, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [
                    conv.conv2d(
                        adtens, bdtens, aivec_val, bivec_val, border_mode="valid"
                    )
                ],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val, border_mode="full")],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        aivec_val = [6, 2, 8, 3]
        bivec_val = [4, 2, 5, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [
                    conv.conv2d(
                        adtens, bdtens, aivec_val, bivec_val, border_mode="valid"
                    )
                ],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val, border_mode="full")],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        aivec_val = [3, 6, 7, 5]
        bivec_val = [5, 6, 3, 2]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [
                    conv.conv2d(
                        adtens, bdtens, aivec_val, bivec_val, border_mode="valid"
                    )
                ],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val, border_mode="full")],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        aivec_val = [3, 6, 7, 5]
        bivec_val = [5, 6, 2, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [
                    conv.conv2d(
                        adtens, bdtens, aivec_val, bivec_val, border_mode="valid"
                    )
                ],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val, border_mode="full")],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        aivec_val = [5, 2, 4, 3]
        bivec_val = [6, 2, 4, 3]
        adtens_val = rand(*aivec_val)
        bdtens_val = rand(*bivec_val)
        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [
                    conv.conv2d(
                        adtens, bdtens, aivec_val, bivec_val, border_mode="valid"
                    )
                ],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )

        with pytest.warns(DeprecationWarning):
            self._compile_and_check(
                [adtens, bdtens],
                [conv.conv2d(adtens, bdtens, aivec_val, bivec_val, border_mode="full")],
                [adtens_val, bdtens_val],
                conv.ConvOp,
                excluding=["conv_gemm"],
            )


# Test that broadcasting of gradients works correctly when using the
# nnet.conv2d() interface. This was reported in #3763, and uses the example
# code from that ticket.
def test_broadcast_grad():
    x1 = tensor4("x")
    sigma = scalar("sigma")
    window_radius = 3

    filter_1d = at.arange(-window_radius, window_radius + 1)
    filter_1d = filter_1d.astype(aesara.config.floatX)
    filter_1d = exp(-0.5 * filter_1d**2 / sigma**2)
    filter_1d = filter_1d / filter_1d.sum()

    filter_W = filter_1d.dimshuffle(["x", "x", 0, "x"])

    y = conv2d(x1, filter_W, border_mode="full", filter_shape=[1, 1, None, None])
    # TODO FIXME: Make this a real test and `assert` something
    aesara.grad(y.sum(), sigma)

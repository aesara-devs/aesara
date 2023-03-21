import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.tensor.nnet import corr3d
from aesara.tensor.type import dmatrix, dtensor3, dtensor4, dtensor5, tensor5, vector
from tests import unittest_tools as utt
from tests.tensor.nnet.test_abstract_conv import TestGroupedConv3dNoOptim


@pytest.mark.skipif(
    aesara.config.cxx == "",
    reason="SciPy and cxx needed",
)
class TestCorr3D(utt.InferShapeTester):
    if aesara.config.mode == "FAST_COMPILE":
        mode = aesara.compile.get_mode("FAST_RUN")
    else:
        mode = None
    dtype = aesara.config.floatX

    def setup_method(self):
        self.input = tensor5("input", dtype=self.dtype)
        self.input.name = "default_V"
        self.filters = tensor5("filters", dtype=self.dtype)
        self.filters.name = "default_filters"
        # This tests can run even when aesara.config.blas__ldflags is empty.
        super().setup_method()

    def validate(
        self,
        image_shape,
        filter_shape,
        border_mode="valid",
        subsample=(1, 1, 1),
        input=None,
        filters=None,
        verify_grad=True,
        non_contiguous=False,
        filter_dilation=(1, 1, 1),
    ):
        """
        :param image_shape: The constant shape info passed to corr3dMM.
        :param filter_shape: The constant shape info passed to corr3dMM.
        """
        if not aesara.config.cxx:
            pytest.skip("Need cxx for this test")

        N_image_shape = [
            at.get_scalar_constant_value(at.as_tensor_variable(x)) for x in image_shape
        ]
        N_filter_shape = [
            at.get_scalar_constant_value(at.as_tensor_variable(x)) for x in filter_shape
        ]

        if input is None:
            input = self.input
        if filters is None:
            filters = self.filters

        # AESARA IMPLEMENTATION

        # we create a symbolic function so that verify_grad can work
        def sym_Corr3dMM(input, filters):
            # define aesara graph and function
            input.name = "input"
            filters.name = "filters"
            rval = corr3d.Corr3dMM(border_mode, subsample, filter_dilation)(
                input, filters
            )
            rval.name = "corr_output"
            return rval

        output = sym_Corr3dMM(input, filters)
        output.name = f"Corr3dMM()({input.name},{filters.name})"
        aesara_corr = aesara.function([input, filters], output, mode=self.mode)

        # initialize input and compute result
        rng = np.random.default_rng(28483)

        image_data = rng.random(N_image_shape).astype(self.dtype)
        filter_data = rng.random(N_filter_shape).astype(self.dtype)
        image_data /= 10
        filter_data /= 10
        if non_contiguous:
            image_data = np.transpose(image_data, axes=(0, 1, 4, 3, 2))
            image_data = image_data.copy()
            image_data = np.transpose(image_data, axes=(0, 1, 4, 3, 2))
            filter_data = np.transpose(filter_data, axes=(0, 1, 4, 3, 2))
            filter_data = filter_data.copy()
            filter_data = np.transpose(filter_data, axes=(0, 1, 4, 3, 2))
            assert not image_data.flags["CONTIGUOUS"]
            assert not filter_data.flags["CONTIGUOUS"]

        aesara_output = aesara_corr(image_data, filter_data)

        # REFERENCE IMPLEMENTATION
        # Testing correlation, not convolution. Reverse filters.
        filter_data_corr = np.array(
            filter_data[:, :, ::-1, ::-1, ::-1], copy=True, order="C"
        )
        orig_image_data = image_data
        img_shape3d = np.array(N_image_shape[-3:])
        fil_shape3d = np.array(N_filter_shape[-3:])
        dil_shape3d = np.array(filter_dilation)
        dil_fil_shape3d = (fil_shape3d - 1) * dil_shape3d + 1
        subsample3d = np.array(subsample)
        if border_mode == "full":
            padHWD = dil_fil_shape3d - 1
        elif border_mode == "valid":
            padHWD = np.array([0, 0, 0])
        elif border_mode == "half":
            padHWD = np.floor(dil_fil_shape3d / 2).astype("int32")
        elif isinstance(border_mode, tuple):
            padHWD = np.array(border_mode)
        elif isinstance(border_mode, int):
            padHWD = np.array([border_mode, border_mode, border_mode])
        else:
            raise NotImplementedError(f"Unsupported border_mode {border_mode}")
        out_shape3d = (
            np.floor((img_shape3d + 2 * (padHWD) - dil_fil_shape3d) / subsample3d) + 1
        )
        # avoid numpy deprecation
        out_shape3d = out_shape3d.astype("int32")
        out_shape = (N_image_shape[0], N_filter_shape[0]) + tuple(out_shape3d)
        ref_output = np.zeros(out_shape)

        # loop over output feature maps
        ref_output.fill(0)
        image_data2 = np.zeros(
            (
                N_image_shape[0],
                N_image_shape[1],
                N_image_shape[2] + 2 * padHWD[0],
                N_image_shape[3] + 2 * padHWD[1],
                N_image_shape[4] + 2 * padHWD[2],
            )
        )
        image_data2[
            :,
            :,
            padHWD[0] : padHWD[0] + N_image_shape[2],
            padHWD[1] : padHWD[1] + N_image_shape[3],
            padHWD[2] : padHWD[2] + N_image_shape[4],
        ] = image_data
        image_data = image_data2
        N_image_shape = image_data.shape
        for bb in range(N_image_shape[0]):
            for nn in range(N_filter_shape[0]):
                for im0 in range(N_image_shape[1]):
                    filter3d = filter_data_corr[nn, im0, :, :, :]
                    image3d = image_data[bb, im0, :, :, :]
                    for row in range(ref_output.shape[2]):
                        irow = row * subsample[0]  # image row
                        for col in range(ref_output.shape[3]):
                            icol = col * subsample[1]  # image col
                            for slc in range(ref_output.shape[4]):
                                islc = slc * subsample[2]  # image slice
                                ref_output[bb, nn, row, col, slc] += (
                                    image3d[
                                        irow : irow
                                        + dil_fil_shape3d[0] : filter_dilation[0],
                                        icol : icol
                                        + dil_fil_shape3d[1] : filter_dilation[1],
                                        islc : islc
                                        + dil_fil_shape3d[2] : filter_dilation[2],
                                    ]
                                    * filter3d[::-1, ::-1, ::-1]
                                ).sum()

        utt.assert_allclose(aesara_output, ref_output)

        # TEST GRADIENT
        if verify_grad:
            utt.verify_grad(
                sym_Corr3dMM, [orig_image_data, filter_data], mode=self.mode
            )

    @pytest.mark.slow
    def test_basic(self):
        # Tests that basic correlations work for odd and even
        # dimensions of image and filter shapes, as well as rectangular
        # images and filters.
        border_modes = [
            "valid",
            "full",
            "half",
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (3, 3, 3),
            1,
        ]
        img_shapes = [
            (2, 2, 3, 3, 3),
            (3, 2, 8, 8, 8),
            (3, 2, 7, 5, 5),
            (3, 2, 7, 5, 5),
            (1, 2, 8, 8, 8),
            (1, 2, 7, 5, 5),
        ]
        fil_shapes = [
            (2, 2, 2, 2, 2),
            (1, 2, 5, 5, 5),
            (2, 2, 2, 3, 2),
            (2, 2, 3, 2, 2),
            (1, 2, 5, 5, 5),
            (1, 2, 2, 3, 3),
        ]

        for border_mode in border_modes:
            for img, fil in zip(img_shapes, fil_shapes):
                self.validate(img, fil, border_mode, verify_grad=False)

        # Very slow on with 'full' or 'half'
        self.validate((1, 2, 53, 29, 11), (13, 2, 12, 1, 1), "valid", verify_grad=False)

    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3, 3), (1, 2, 3, 3, 3), "full")
        self.validate((3, 2, 3, 3, 3), (1, 2, 3, 3, 3), "valid")
        self.validate((3, 2, 3, 3, 3), (1, 2, 3, 3, 3), "half")
        self.validate((3, 2, 3, 3, 3), (1, 2, 3, 3, 3), (1, 1, 1))
        self.validate((3, 2, 3, 3, 3), (1, 2, 3, 3, 3), 1)

    @pytest.mark.slow
    def test_subsample(self):
        # Tests correlation where subsampling != (1,1,1)
        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "valid", subsample=(2, 2, 2))
        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "valid", subsample=(2, 1, 1))
        self.validate((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), "valid", subsample=(3, 3, 3))

        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "full", subsample=(2, 2, 2))
        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "full", subsample=(2, 1, 1))
        self.validate((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), "full", subsample=(3, 3, 3))

        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "half", subsample=(2, 2, 2))
        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "half", subsample=(2, 1, 1))
        self.validate((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), "half", subsample=(3, 3, 3))

        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), (1, 1, 1), subsample=(2, 2, 2))
        self.validate((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), (2, 1, 1), subsample=(2, 1, 1))
        self.validate((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), (1, 2, 2), subsample=(3, 3, 3))

        self.validate((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), 1, subsample=(3, 3, 3))

    # Tests correlation where filter dilation != (1,1,1)
    @pytest.mark.parametrize(
        "image_shape, filter_shape, border_mode, filter_dilation",
        [
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "valid", (2, 2, 2)),
            ((3, 2, 14, 10, 10), (2, 2, 2, 3, 3), "valid", (3, 1, 1)),
            ((1, 1, 14, 14, 14), (1, 1, 3, 3, 3), "valid", (2, 3, 3)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "full", (2, 2, 2)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "full", (3, 1, 1)),
            ((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), "full", (2, 3, 3)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "half", (2, 2, 2)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), "half", (3, 1, 1)),
            ((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), "half", (2, 3, 3)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), (1, 1, 1), (2, 2, 2)),
            ((3, 2, 7, 5, 5), (2, 2, 2, 3, 3), (2, 1, 1), (2, 1, 1)),
            ((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), (1, 2, 1), (1, 2, 1)),
            ((1, 1, 6, 6, 6), (1, 1, 3, 3, 3), (1, 1, 2), (1, 1, 2)),
        ],
    )
    def test_filter_dilation(
        self, image_shape, filter_shape, border_mode, filter_dilation
    ):
        self.validate(
            image_shape, filter_shape, border_mode, filter_dilation=filter_dilation
        )

    def test_filter_dilation_subsample(self):
        self.validate(
            (1, 1, 6, 6, 6),
            (1, 1, 3, 3, 3),
            1,
            subsample=(3, 3, 3),
            filter_dilation=(2, 2, 2),
        )

    @pytest.mark.parametrize(
        "border_mode",
        [
            "valid",
            "full",
            "half",
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (3, 3, 3),
            1,
        ],
    )
    def test_shape_Constant_tensor(self, border_mode):
        # Tests correlation where the {image,filter}_shape is a Constant tensor
        as_t = at.as_tensor_variable
        self.validate(
            (as_t(3), as_t(2), as_t(7), as_t(5), as_t(5)), (5, 2, 2, 3, 3), border_mode
        )
        self.validate(as_t([3, 2, 7, 5, 5]), (5, 2, 2, 3, 3), border_mode)
        self.validate(as_t((3, 2, 7, 5, 5)), (5, 2, 2, 3, 3), border_mode)
        self.validate(
            (3, 2, 7, 5, 5), (as_t(5), as_t(2), as_t(2), as_t(3), as_t(3)), "valid"
        )
        self.validate((3, 2, 7, 5, 5), as_t([5, 2, 2, 3, 3]), border_mode)
        self.validate(as_t([3, 2, 7, 5, 5]), as_t([5, 2, 2, 3, 3]), border_mode)

    def test_invalid_filter_shape(self):
        # Tests scenario where filter_shape[1] != input_shape[1]
        with pytest.raises(ValueError):
            self.validate((3, 2, 8, 8, 8), (4, 3, 5, 5, 8), "valid")

    def test_full_mode(self):
        # Tests basic correlation in full mode and case where filter
        # is larger than the input image.
        self.validate((3, 1, 4, 4, 4), (2, 1, 5, 5, 5), "full")

        def f():
            self.validate((3, 2, 5, 5, 5), (4, 2, 8, 8, 8), "valid")

        with pytest.raises(Exception):
            f()

    def test_wrong_input(self):
        # Make sure errors are raised when image and kernel are not 5D tensors
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8, 8), (4, 2, 5, 5, 5), "valid", input=dmatrix())
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8, 8), (4, 2, 5, 5, 5), "valid", input=vector())
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8, 8), (4, 2, 5, 5, 5), "valid", input=dtensor3())
        with pytest.raises(Exception):
            self.validate((3, 2, 8, 8, 8), (4, 2, 5, 5, 5), "valid", input=dtensor4())

    @pytest.mark.skipif(not aesara.config.cxx, reason="Need cxx for this test")
    def test_dtype_upcast(self):
        # Checks dtype upcast for Corr3dMM methods.

        rng = np.random.default_rng(28483)

        def rand(shape, dtype="float64"):
            r = np.asarray(rng.random(shape), dtype=dtype)
            return r * 2 - 1

        ops = [corr3d.Corr3dMM, corr3d.Corr3dMMGradWeights, corr3d.Corr3dMMGradInputs]
        a_shapes = [[4, 5, 6, 3, 3], [1, 5, 6, 3, 3], [1, 5, 6, 3, 3]]
        b_shapes = [[7, 5, 3, 2, 2], [1, 5, 3, 1, 1], [7, 1, 3, 1, 1]]
        dtypes = ["float32", "float64"]

        for op, a_shape, b_shape in zip(ops, a_shapes, b_shapes):
            for a_dtype in dtypes:
                for b_dtype in dtypes:
                    c_dtype = aesara.scalar.upcast(a_dtype, b_dtype)
                    a_tens = tensor5(dtype=a_dtype)
                    b_tens = tensor5(dtype=b_dtype)
                    a_tens_val = rand(a_shape, dtype=a_dtype)
                    b_tens_val = rand(b_shape, dtype=b_dtype)

                    c_tens = op()(a_tens, b_tens)
                    f = aesara.function([a_tens, b_tens], c_tens, mode=self.mode)
                    assert f(a_tens_val, b_tens_val).dtype == c_dtype

    @pytest.mark.slow
    @pytest.mark.skipif(
        aesara.config.mode == "FAST_COMPILE" or not aesara.config.cxx,
        reason="Need cxx for this test",
    )
    def test_infer_shape_forward(self):
        rng = np.random.default_rng(28483)

        def rand(*shape):
            r = np.asarray(rng.random(shape), dtype="float64")
            return r * 2 - 1

        corr3dMM = corr3d.Corr3dMM

        adtens = dtensor5()
        bdtens = dtensor5()
        aivec_vals = [
            [4, 5, 6, 3, 3],
            [6, 2, 8, 3, 3],
            [3, 6, 7, 5, 5],
            [3, 6, 7, 5, 5],
            [5, 2, 4, 3, 3],
        ]
        bivec_vals = [
            [7, 5, 3, 2, 2],
            [4, 2, 5, 3, 3],
            [5, 6, 3, 2, 2],
            [5, 6, 2, 3, 3],
            [6, 2, 4, 3, 3],
        ]
        modes = ["valid", "full", "half", (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), 1]
        subsamples = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # Corr3dMM
                    cdtens = corr3dMM(border_mode=mode, subsample=subsample)(
                        adtens, bdtens
                    )
                    self._compile_and_check(
                        [adtens, bdtens],
                        [cdtens],
                        [adtens_val, bdtens_val],
                        corr3dMM,
                        warn=False,
                    )

    @pytest.mark.slow
    @pytest.mark.skipif(
        aesara.config.mode == "FAST_COMPILE" or not aesara.config.cxx,
        reason="Need cxx for this test",
    )
    def test_infer_shape_gradW(self):
        rng = np.random.default_rng(28483)

        def rand(*shape):
            r = np.asarray(rng.random(shape), dtype="float64")
            return r * 2 - 1

        corr3dMM = corr3d.Corr3dMM
        gradW = corr3d.Corr3dMMGradWeights

        adtens = dtensor5()
        bdtens = dtensor5()
        aivec_vals = [
            [1, 5, 6, 3, 3],
            [8, 2, 7, 3, 3],
            [1, 6, 9, 4, 4],
            [9, 6, 8, 5, 5],
            [9, 1, 6, 8, 8],
        ]
        bivec_vals = [
            [7, 5, 3, 1, 1],
            [4, 2, 5, 3, 3],
            [12, 6, 3, 2, 2],
            [5, 6, 1, 3, 3],
            [11, 1, 3, 3, 3],
        ]
        modes = ["valid", "full", "half", (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), 1]
        subsamples = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # Corr3dMM
                    cdtens = corr3dMM(border_mode=mode, subsample=subsample)(
                        adtens, bdtens
                    )
                    f = aesara.function([adtens, bdtens], cdtens)
                    cdtens_val = f(adtens_val, bdtens_val)
                    # Corr3dMM_gradWeights
                    shape = (
                        aesara.shared(bivec_val[2]),
                        aesara.shared(bivec_val[3]),
                        aesara.shared(bivec_val[4]),
                    )
                    bdtens_g = gradW(border_mode=mode, subsample=subsample)(
                        adtens, cdtens, shape=shape
                    )
                    self._compile_and_check(
                        [adtens, cdtens],
                        [bdtens_g],
                        [adtens_val, cdtens_val],
                        gradW,
                        warn=False,
                    )

    @pytest.mark.slow
    @pytest.mark.skipif(
        aesara.config.mode == "FAST_COMPILE" or not aesara.config.cxx,
        reason="Need cxx for this test",
    )
    def test_infer_shape_gradI(self):
        rng = np.random.default_rng(28483)

        def rand(*shape):
            r = np.asarray(rng.random(shape), dtype="float64")
            return r * 2 - 1

        corr3dMM = corr3d.Corr3dMM
        gradI = corr3d.Corr3dMMGradInputs

        adtens = dtensor5()
        bdtens = dtensor5()
        aivec_vals = [
            [1, 5, 6, 3, 3],
            [8, 2, 7, 3, 3],
            [1, 6, 9, 4, 4],
            [9, 6, 8, 5, 5],
            [9, 1, 6, 8, 8],
        ]
        bivec_vals = [
            [7, 5, 3, 1, 1],
            [4, 2, 5, 3, 3],
            [12, 6, 3, 2, 2],
            [5, 6, 1, 3, 3],
            [7, 1, 3, 4, 4],
        ]
        modes = ["valid", "full", "half", (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), 1]
        subsamples = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

        for aivec_val, bivec_val in zip(aivec_vals, bivec_vals):
            adtens_val = rand(*aivec_val)
            bdtens_val = rand(*bivec_val)
            for mode in modes:
                for subsample in subsamples:
                    # Corr3dMM
                    cdtens = corr3dMM(border_mode=mode, subsample=subsample)(
                        adtens, bdtens
                    )
                    f = aesara.function([adtens, bdtens], cdtens)
                    cdtens_val = f(adtens_val, bdtens_val)
                    # Corr3dMM_gradInputs
                    shape = (
                        aesara.shared(aivec_val[2]),
                        aesara.shared(aivec_val[3]),
                        aesara.shared(aivec_val[4]),
                    )
                    adtens_g = gradI(border_mode=mode, subsample=subsample)(
                        bdtens, cdtens, shape=shape
                    )
                    self._compile_and_check(
                        [bdtens, cdtens],
                        [adtens_g],
                        [bdtens_val, cdtens_val],
                        gradI,
                        warn=False,
                    )

    def test_non_contiguous(self):
        self.validate((2, 2, 3, 3, 3), (2, 2, 2, 2, 2), "valid", non_contiguous=True)
        self.validate((3, 2, 8, 8, 8), (2, 2, 5, 5, 5), "valid", non_contiguous=True)
        self.validate((3, 2, 7, 5, 5), (3, 2, 2, 3, 3), "valid", non_contiguous=True)
        self.validate((3, 2, 7, 5, 5), (3, 2, 3, 2, 2), "valid", non_contiguous=True)
        self.validate((3, 1, 8, 8, 8), (2, 1, 5, 5, 5), "full", non_contiguous=True)
        self.validate((3, 1, 8, 8, 8), (2, 1, 5, 5, 5), "half", non_contiguous=True)
        self.validate((3, 1, 8, 8, 8), (2, 1, 5, 5, 5), (1, 1, 1), non_contiguous=True)
        self.validate((3, 1, 7, 5, 5), (2, 1, 2, 3, 3), (1, 1, 2), non_contiguous=True)
        self.validate((3, 1, 7, 5, 5), (2, 1, 2, 3, 3), (1, 2, 1), non_contiguous=True)
        self.validate((3, 1, 7, 5, 5), (2, 1, 2, 3, 3), (2, 1, 1), non_contiguous=True)


class TestGroupCorr3d(TestGroupedConv3dNoOptim):
    mode = aesara.compile.get_mode("FAST_RUN")
    conv_op = corr3d.Corr3dMM
    conv_gradw_op = corr3d.Corr3dMMGradWeights
    conv_gradi_op = corr3d.Corr3dMMGradInputs
    flip_filter = True
    is_dnn = False

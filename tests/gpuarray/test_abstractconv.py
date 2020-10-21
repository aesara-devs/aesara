import numpy as np
import pytest


pygpu = pytest.importorskip("pygpu")
gpuarray = pygpu.gpuarray

from tests.gpuarray.config import mode_with_gpu, test_ctx_name
from tests.tensor.nnet.test_abstract_conv import (
    BaseTestConv2d,
    BaseTestConv3d,
    TestConv2dTranspose,
    TestConvTypes,
)
from theano.gpuarray.blas import (
    GpuCorr3dMM,
    GpuCorr3dMM_gradInputs,
    GpuCorr3dMM_gradWeights,
    GpuCorrMM,
    GpuCorrMM_gradInputs,
    GpuCorrMM_gradWeights,
)
from theano.gpuarray.dnn import (
    GpuDnnConv,
    GpuDnnConvGradI,
    GpuDnnConvGradW,
    dnn_available,
)
from theano.gpuarray.type import GpuArrayType, get_context, gpuarray_shared_constructor


gpu_ftensor4 = GpuArrayType(dtype="float32", broadcastable=(False,) * 4)


class TestDnnConv2d(BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.shared = staticmethod(gpuarray_shared_constructor)
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]

    def run_test_case(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        if not dnn_available(test_ctx_name):
            pytest.skip(dnn_available.msg)

        mode = mode_with_gpu

        if fd != (1, 1):
            pytest.skip("Doesn't have CUDNN implementation")

        o = self.get_output_shape(i, f, s, b, fd)

        self.run_fwd(
            inputs_shape=i,
            filters_shape=f,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConv,
        )
        self.run_gradweight(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConvGradW,
        )
        self.run_gradinput(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConvGradI,
        )

    def run_test_case_gi(
        self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False
    ):
        if not dnn_available(test_ctx_name):
            pytest.skip(dnn_available.msg)

        if fd != (1, 1):
            pytest.skip("Doesn't have CUDNN implementation")

        mode = mode_with_gpu

        if not expect_error:
            self.run_gradinput(
                inputs_shape=i,
                filters_shape=f,
                output_shape=o,
                subsample=s,
                verify_grad=True,
                mode=mode,
                provide_shape=provide_shape,
                border_mode=b,
                filter_flip=flip,
                target_op=GpuDnnConvGradI,
                filter_dilation=fd,
            )
        else:
            with pytest.raises((RuntimeError, ValueError)):
                self.run_gradinput(
                    inputs_shape=i,
                    filters_shape=f,
                    output_shape=o,
                    subsample=s,
                    verify_grad=False,
                    mode=mode,
                    provide_shape=provide_shape,
                    border_mode=b,
                    filter_flip=flip,
                    target_op=GpuDnnConvGradI,
                    ref=None,
                    filter_dilation=fd,
                )


class TestDnnConv3d(BaseTestConv3d):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.shared = staticmethod(gpuarray_shared_constructor)
        # provide_shape is not used by the cuDNN impementation
        cls.provide_shape = [False]

    def run_test_case(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        if not dnn_available(test_ctx_name):
            pytest.skip(dnn_available.msg)

        mode = mode_with_gpu

        if fd != (1, 1, 1):
            pytest.skip("Doesn't have CUDNN implementation")

        o = self.get_output_shape(i, f, s, b, fd)

        self.run_fwd(
            inputs_shape=i,
            filters_shape=f,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConv,
        )
        self.run_gradweight(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConvGradW,
        )
        self.run_gradinput(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuDnnConvGradI,
        )

    def run_test_case_gi(
        self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False
    ):
        if not dnn_available(test_ctx_name):
            pytest.skip(dnn_available.msg)

        if fd != (1, 1, 1):
            pytest.skip("Doesn't have CUDNN implementation")

        mode = mode_with_gpu

        if not expect_error:
            self.run_gradinput(
                inputs_shape=i,
                filters_shape=f,
                output_shape=o,
                subsample=s,
                verify_grad=True,
                mode=mode,
                provide_shape=provide_shape,
                border_mode=b,
                filter_flip=flip,
                target_op=GpuDnnConvGradI,
                filter_dilation=fd,
            )
        else:
            with pytest.raises((RuntimeError, ValueError)):
                self.run_gradinput(
                    inputs_shape=i,
                    filters_shape=f,
                    output_shape=o,
                    subsample=s,
                    verify_grad=False,
                    mode=mode,
                    provide_shape=provide_shape,
                    border_mode=b,
                    filter_flip=flip,
                    target_op=GpuDnnConvGradI,
                    ref=None,
                    filter_dilation=fd,
                )


class TestCorrMMConv2d(BaseTestConv2d):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.shared = staticmethod(gpuarray_shared_constructor)
        cls.mode = mode_with_gpu.excluding("cudnn")

    def run_test_case(self, i, f, s, b, flip, provide_shape, fd=(1, 1)):
        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(
            inputs_shape=i,
            filters_shape=f,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=(GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs),
            filter_dilation=fd,
        )
        self.run_gradweight(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuCorrMM_gradWeights,
            filter_dilation=fd,
        )
        self.run_gradinput(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuCorrMM_gradInputs,
            filter_dilation=fd,
        )

    def run_test_case_gi(
        self, i, f, o, s, b, flip, provide_shape, fd=(1, 1), expect_error=False
    ):
        mode = self.mode
        if not expect_error:
            self.run_gradinput(
                inputs_shape=i,
                filters_shape=f,
                output_shape=o,
                subsample=s,
                verify_grad=True,
                mode=mode,
                provide_shape=provide_shape,
                border_mode=b,
                filter_flip=flip,
                target_op=GpuCorrMM_gradInputs,
                filter_dilation=fd,
            )
        else:
            with pytest.raises(ValueError):
                self.run_gradinput(
                    inputs_shape=i,
                    filters_shape=f,
                    output_shape=o,
                    subsample=s,
                    verify_grad=False,
                    mode=mode,
                    provide_shape=provide_shape,
                    border_mode=b,
                    filter_flip=flip,
                    target_op=GpuCorrMM_gradInputs,
                    ref=None,
                    filter_dilation=fd,
                )


class TestCorrMMConv3d(BaseTestConv3d):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.shared = staticmethod(gpuarray_shared_constructor)
        cls.mode = mode_with_gpu.excluding("cudnn")

    def run_test_case(self, i, f, s, b, flip, provide_shape, fd=(1, 1, 1)):
        mode = self.mode
        o = self.get_output_shape(i, f, s, b, fd)
        self.run_fwd(
            inputs_shape=i,
            filters_shape=f,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=(GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs),
            filter_dilation=fd,
        )
        self.run_gradweight(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuCorr3dMM_gradWeights,
            filter_dilation=fd,
        )
        self.run_gradinput(
            inputs_shape=i,
            filters_shape=f,
            output_shape=o,
            subsample=s,
            verify_grad=True,
            mode=mode,
            provide_shape=provide_shape,
            border_mode=b,
            filter_flip=flip,
            target_op=GpuCorr3dMM_gradInputs,
            filter_dilation=fd,
        )

    def run_test_case_gi(
        self, i, f, o, s, b, flip, provide_shape, fd=(1, 1, 1), expect_error=False
    ):
        mode = self.mode
        if not expect_error:
            self.run_gradinput(
                inputs_shape=i,
                filters_shape=f,
                output_shape=o,
                subsample=s,
                verify_grad=True,
                mode=mode,
                provide_shape=provide_shape,
                border_mode=b,
                filter_flip=flip,
                target_op=GpuCorr3dMM_gradInputs,
                filter_dilation=fd,
            )
        else:
            with pytest.raises(ValueError):
                self.run_gradinput(
                    inputs_shape=i,
                    filters_shape=f,
                    output_shape=o,
                    subsample=s,
                    verify_grad=False,
                    mode=mode,
                    provide_shape=provide_shape,
                    border_mode=b,
                    filter_flip=flip,
                    target_op=GpuCorr3dMM_gradInputs,
                    ref=None,
                    filter_dilation=fd,
                )


class TestDnnConvTypes(TestConvTypes):
    def setup_method(self):
        self.input = gpu_ftensor4()
        self.filters = gpu_ftensor4()
        self.topgrad = gpu_ftensor4()
        self.constant_tensor = gpuarray.array(
            np.zeros((3, 5, 7, 11), dtype="float32"), context=get_context(test_ctx_name)
        )
        super().setup_method()


class TestConv2dTranspose(TestConv2dTranspose):
    mode = mode_with_gpu

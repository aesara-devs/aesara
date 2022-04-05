import numpy as np
import pytest

import aesara
import aesara.gpuarray.fft
from aesara.gpuarray.fft import pycuda_available, pygpu_available, skcuda_available
from aesara.tensor.type import matrix
from tests import unittest_tools as utt
from tests.gpuarray.config import mode_with_gpu


# Skip tests if pygpu is not available.
if not pygpu_available:  # noqa
    pytest.skip("Optional package pygpu not available", allow_module_level=True)
if not skcuda_available:  # noqa
    pytest.skip("Optional package scikit-cuda not available", allow_module_level=True)
if not pycuda_available:  # noqa
    pytest.skip("Optional package pycuda not available", allow_module_level=True)

# Transform sizes
N = 32


class TestFFT:
    def test_1Dfft(self):
        inputs_val = np.random.random((1, N)).astype("float32")

        x = matrix("x", dtype="float32")
        rfft = aesara.gpuarray.fft.curfft(x)
        f_rfft = aesara.function([x], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft(inputs_val)
        res_rfft_comp = np.asarray(res_rfft[:, :, 0]) + 1j * np.asarray(
            res_rfft[:, :, 1]
        )

        rfft_ref = np.fft.rfft(inputs_val, axis=1)

        utt.assert_allclose(rfft_ref, res_rfft_comp)

        m = rfft.type()
        irfft = aesara.gpuarray.fft.cuirfft(m)
        f_irfft = aesara.function([m], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft(res_rfft)

        utt.assert_allclose(inputs_val, np.asarray(res_irfft))

        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return aesara.gpuarray.fft.curfft(inp)

        inputs_val = np.random.random((1, N)).astype("float32")
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return aesara.gpuarray.fft.cuirfft(inp)

        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype("float32")
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_rfft(self):
        inputs_val = np.random.random((1, N, N)).astype("float32")
        inputs = aesara.shared(inputs_val)

        rfft = aesara.gpuarray.fft.curfft(inputs)
        f_rfft = aesara.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = np.asarray(res_rfft[:, :, :, 0]) + 1j * np.asarray(
            res_rfft[:, :, :, 1]
        )

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

    def test_irfft(self):
        inputs_val = np.random.random((1, N, N)).astype("float32")
        inputs = aesara.shared(inputs_val)

        fft = aesara.gpuarray.fft.curfft(inputs)
        f_fft = aesara.function([], fft, mode=mode_with_gpu)
        res_fft = f_fft()

        m = fft.type()
        ifft = aesara.gpuarray.fft.cuirfft(m)
        f_ifft = aesara.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_fft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft))

        inputs_val = np.random.random((1, N, N, 2)).astype("float32")
        inputs = aesara.shared(inputs_val)

        irfft = aesara.gpuarray.fft.cuirfft(inputs)
        f_irfft = aesara.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()
        inputs_ref = inputs_val[..., 0] + inputs_val[..., 1] * 1j

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref, res_irfft, atol=1e-4, rtol=1e-4)

    def test_type(self):
        inputs_val = np.random.random((1, N)).astype("float64")
        inputs = aesara.shared(inputs_val)

        with pytest.raises(AssertionError):
            aesara.gpuarray.fft.curfft(inputs)
        with pytest.raises(AssertionError):
            aesara.gpuarray.fft.cuirfft(inputs)

    def test_norm(self):
        inputs_val = np.random.random((1, N, N)).astype("float32")
        inputs = aesara.shared(inputs_val)

        # Unitary normalization
        rfft = aesara.gpuarray.fft.curfft(inputs, norm="ortho")
        f_rfft = aesara.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = np.asarray(res_rfft[:, :, :, 0]) + 1j * np.asarray(
            res_rfft[:, :, :, 1]
        )

        rfft_ref = np.fft.rfftn(inputs_val, axes=(1, 2))

        utt.assert_allclose(rfft_ref / N, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # No normalization
        rfft = aesara.gpuarray.fft.curfft(inputs, norm="no_norm")
        f_rfft = aesara.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()
        res_rfft_comp = np.asarray(res_rfft[:, :, :, 0]) + 1j * np.asarray(
            res_rfft[:, :, :, 1]
        )

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

        # Inverse FFT inputs
        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype("float32")
        inputs = aesara.shared(inputs_val)
        inputs_ref = inputs_val[:, :, :, 0] + 1j * inputs_val[:, :, :, 1]

        # Unitary normalization inverse FFT
        irfft = aesara.gpuarray.fft.cuirfft(inputs, norm="ortho")
        f_irfft = aesara.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        irfft_ref = np.fft.irfftn(inputs_ref, axes=(1, 2))

        utt.assert_allclose(irfft_ref * N, res_irfft, atol=1e-4, rtol=1e-4)

        # No normalization inverse FFT
        irfft = aesara.gpuarray.fft.cuirfft(inputs, norm="no_norm")
        f_irfft = aesara.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        utt.assert_allclose(irfft_ref * N**2, res_irfft, atol=1e-4, rtol=1e-4)

    def test_grad(self):
        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return aesara.gpuarray.fft.curfft(inp)

        inputs_val = np.random.random((1, N, N)).astype("float32")
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return aesara.gpuarray.fft.cuirfft(inp)

        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype("float32")
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_rfft(inp):
            return aesara.gpuarray.fft.curfft(inp, norm="ortho")

        inputs_val = np.random.random((1, N, N)).astype("float32")
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return aesara.gpuarray.fft.cuirfft(inp, norm="no_norm")

        inputs_val = np.random.random((1, N, N // 2 + 1, 2)).astype("float32")
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_odd(self):
        M = N - 1

        inputs_val = np.random.random((1, M, M)).astype("float32")
        inputs = aesara.shared(inputs_val)

        rfft = aesara.gpuarray.fft.curfft(inputs)
        f_rfft = aesara.function([], rfft, mode=mode_with_gpu)
        res_rfft = f_rfft()

        res_rfft_comp = np.asarray(res_rfft[:, :, :, 0]) + 1j * np.asarray(
            res_rfft[:, :, :, 1]
        )

        rfft_ref = np.fft.rfftn(inputs_val, s=(M, M), axes=(1, 2))

        utt.assert_allclose(rfft_ref, res_rfft_comp, atol=1e-4, rtol=1e-4)

        m = rfft.type()
        ifft = aesara.gpuarray.fft.cuirfft(m, is_odd=True)
        f_ifft = aesara.function([m], ifft, mode=mode_with_gpu)
        res_ifft = f_ifft(res_rfft)

        utt.assert_allclose(inputs_val, np.asarray(res_ifft))

        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype("float32")
        inputs = aesara.shared(inputs_val)

        irfft = aesara.gpuarray.fft.cuirfft(inputs, norm="ortho", is_odd=True)
        f_irfft = aesara.function([], irfft, mode=mode_with_gpu)
        res_irfft = f_irfft()

        inputs_ref = inputs_val[:, :, :, 0] + 1j * inputs_val[:, :, :, 1]
        irfft_ref = np.fft.irfftn(inputs_ref, s=(M, M), axes=(1, 2)) * M

        utt.assert_allclose(irfft_ref, res_irfft, atol=1e-4, rtol=1e-4)

        # The numerical gradient of the FFT is sensitive, must set large
        # enough epsilon to get good accuracy.
        eps = 1e-1

        def f_rfft(inp):
            return aesara.gpuarray.fft.curfft(inp)

        inputs_val = np.random.random((1, M, M)).astype("float32")
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return aesara.gpuarray.fft.cuirfft(inp, is_odd=True)

        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype("float32")
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_rfft(inp):
            return aesara.gpuarray.fft.curfft(inp, norm="ortho")

        inputs_val = np.random.random((1, M, M)).astype("float32")
        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)

        def f_irfft(inp):
            return aesara.gpuarray.fft.cuirfft(inp, norm="no_norm", is_odd=True)

        inputs_val = np.random.random((1, M, M // 2 + 1, 2)).astype("float32")
        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)

    def test_params(self):
        inputs_val = np.random.random((1, N)).astype("float32")
        inputs = aesara.shared(inputs_val)

        with pytest.raises(ValueError):
            aesara.gpuarray.fft.curfft(inputs, norm=123)

        inputs_val = np.random.random((1, N // 2 + 1, 2)).astype("float32")
        inputs = aesara.shared(inputs_val)

        with pytest.raises(ValueError):
            aesara.gpuarray.fft.cuirfft(inputs, norm=123)
        with pytest.raises(ValueError):
            aesara.gpuarray.fft.cuirfft(inputs, is_odd=123)

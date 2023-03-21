import numpy as np
import pytest

import aesara
from aesara.tensor.math import _allclose
from aesara.tensor.signal import conv
from aesara.tensor.type import TensorType, dtensor3, dtensor4, dvector, matrix
from tests import unittest_tools as utt


_ = pytest.importorskip("scipy.signal")


class TestSignalConv2D:
    def validate(self, image_shape, filter_shape, out_dim, verify_grad=True):
        image_dim = len(image_shape)
        filter_dim = len(filter_shape)
        input = TensorType("float64", shape=(None,) * image_dim)()
        filters = TensorType("float64", shape=(None,) * filter_dim)()

        bsize = image_shape[0]
        if image_dim != 3:
            bsize = 1
        nkern = filter_shape[0]
        if filter_dim != 3:
            nkern = 1

        # AESARA IMPLEMENTATION ############
        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            return conv.conv2d(input, filters)

        output = sym_conv2d(input, filters)
        assert output.ndim == out_dim
        aesara_conv = aesara.function([input, filters], output)

        # initialize input and compute result
        image_data = np.random.random(image_shape)
        filter_data = np.random.random(filter_shape)
        aesara_output = aesara_conv(image_data, filter_data)

        # REFERENCE IMPLEMENTATION ############
        out_shape2d = np.array(image_shape[-2:]) - np.array(filter_shape[-2:]) + 1
        ref_output = np.zeros(tuple(out_shape2d))

        # reshape as 3D input tensors to make life easier
        image_data3d = image_data.reshape((bsize,) + image_shape[-2:])
        filter_data3d = filter_data.reshape((nkern,) + filter_shape[-2:])
        # reshape aesara output as 4D to make life easier
        aesara_output4d = aesara_output.reshape(
            (
                bsize,
                nkern,
            )
            + aesara_output.shape[-2:]
        )

        # loop over mini-batches (if required)
        for b in range(bsize):
            # loop over filters (if required)
            for k in range(nkern):
                image2d = image_data3d[b, :, :]
                filter2d = filter_data3d[k, :, :]
                output2d = np.zeros(ref_output.shape)
                for row in range(ref_output.shape[0]):
                    for col in range(ref_output.shape[1]):
                        output2d[row, col] += (
                            image2d[
                                row : row + filter2d.shape[0],
                                col : col + filter2d.shape[1],
                            ]
                            * filter2d[::-1, ::-1]
                        ).sum()

                assert _allclose(aesara_output4d[b, k, :, :], output2d)

        # TEST GRADIENT ############
        if verify_grad:
            utt.verify_grad(sym_conv2d, [image_data, filter_data])

    @pytest.mark.skipif(
        aesara.config.cxx == "",
        reason="conv2d tests need a c++ compiler",
    )
    def test_basic(self):
        # Basic functionality of nnet.conv.ConvOp is already tested by
        # its own test suite.  We just have to test whether or not
        # signal.conv.conv2d can support inputs and filters of type
        # matrix or tensor3.
        self.validate((1, 4, 5), (2, 2, 3), out_dim=4, verify_grad=True)
        self.validate((7, 5), (5, 2, 3), out_dim=3, verify_grad=False)
        self.validate((3, 7, 5), (2, 3), out_dim=3, verify_grad=False)
        self.validate((7, 5), (2, 3), out_dim=2, verify_grad=False)

    def test_fail(self):
        # Test that conv2d fails for dimensions other than 2 or 3.

        with pytest.raises(Exception):
            conv.conv2d(dtensor4(), dtensor3())
        with pytest.raises(Exception):
            conv.conv2d(dtensor3(), dvector())

    def test_bug_josh_reported(self):
        # Test refers to a bug reported by Josh, when due to a bad merge these
        # few lines of code failed. See
        # http://groups.google.com/group/theano-dev/browse_thread/thread/8856e7ca5035eecb

        m1 = matrix()
        m2 = matrix()
        conv.conv2d(m1, m2)

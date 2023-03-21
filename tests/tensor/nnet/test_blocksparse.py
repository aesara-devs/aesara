"""
    Tests for block sparse dot
"""
import numpy as np

import aesara
import aesara.tensor as at
import tests.unittest_tools as utt
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.nnet.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_dot,
    sparse_block_gemv,
    sparse_block_outer,
)
from aesara.tensor.type import fmatrix, ftensor3, ftensor4, imatrix


class TestBlockSparseGemvAndOuter(utt.InferShapeTester):
    def setup_method(self):
        mode = None
        if aesara.config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        self.mode = aesara.compile.get_mode(mode).excluding("constant_folding")
        self.gemv_op = sparse_block_gemv
        self.outer_op = sparse_block_outer
        self.gemv_class = SparseBlockGemv
        self.outer_class = SparseBlockOuter
        super().setup_method()

    @staticmethod
    def gemv_data():
        nInputBlock = 8
        nOutputBlock = 7
        inputSize = 6
        outputSize = 5
        inputWindowSize = 4
        outputWindowSize = 3
        batchSize = 2

        rng = np.random.default_rng(230920)

        input = rng.standard_normal((batchSize, inputWindowSize, inputSize)).astype(
            "float32"
        )
        inputIndice = np.vstack(
            rng.permutation(nInputBlock)[:inputWindowSize] for _ in range(batchSize)
        ).astype("int32")
        outputIndice = np.vstack(
            rng.permutation(nOutputBlock)[:outputWindowSize] for _ in range(batchSize)
        ).astype("int32")
        weight = rng.standard_normal(
            (nInputBlock, nOutputBlock, inputSize, outputSize)
        ).astype("float32")
        bias = rng.standard_normal((nOutputBlock, outputSize)).astype("float32")

        return weight, input, inputIndice, bias, outputIndice

    @staticmethod
    def outer_data():
        nInputBlock = 8
        nOutputBlock = 7
        xSize = 6
        ySize = 5
        xWindowSize = 4
        yWindowSize = 3
        batchSize = 2

        rng = np.random.default_rng(230920)

        o = rng.standard_normal((nInputBlock, nOutputBlock, xSize, ySize)).astype(
            "float32"
        )
        x = rng.standard_normal((batchSize, xWindowSize, xSize)).astype("float32")
        y = rng.standard_normal((batchSize, yWindowSize, ySize)).astype("float32")
        xIdx = np.vstack(
            rng.integers(0, nInputBlock, size=xWindowSize) for _ in range(batchSize)
        ).astype("int32")
        yIdx = np.vstack(
            rng.integers(0, nOutputBlock, size=yWindowSize) for _ in range(batchSize)
        ).astype("int32")

        return o, x, y, xIdx, yIdx

    @staticmethod
    def gemv_numpy(o, W, h, iIdx, oIdx):
        for b in range(o.shape[0]):
            for j in range(o.shape[1]):
                outputIdx = oIdx[b, j]
                for i in range(h.shape[1]):
                    inputIdx = iIdx[b, i]
                    w = W[inputIdx, outputIdx]
                    o[b, j, :] += np.dot(h[b, i], w)
        return o

    @staticmethod
    def gemv_numpy2(o, W, h, iIdx, oIdx):
        """
        Other implementation
        """
        from numpy import ix_

        for b in range(o.shape[0]):
            w = W[ix_(iIdx[b], oIdx[b])].swapaxes(1, 2)
            w = w.reshape((w.shape[0] * w.shape[1], w.shape[2] * w.shape[3]))
            o[b] += np.dot(h[b].ravel(), w).reshape(o.shape[1:])
        return o

    @staticmethod
    def gemv_numpy3(o, W, h, iIdx, oIdx):
        """
        Other implementation
        """
        from numpy import ix_

        for b in range(o.shape[0]):
            w = W[ix_(iIdx[b], oIdx[b])]
            # The next three lines do the same operation. The last one is the
            # fastest
            # o[b] += (h[b][:, None, :, None] * w).sum(axis=(0, 2))
            # o[b] += np.tensordot(h[b], w, [(0,1),(0,2)])
            o[b] += np.einsum("ik,ijkl", h[b], w)
        return o

    @staticmethod
    def outer_numpy(o, x, y, xIdx, yIdx):
        for b in range(x.shape[0]):
            for i in range(xIdx.shape[1]):
                for j in range(yIdx.shape[1]):
                    o[xIdx[b, i], yIdx[b, j]] += np.outer(x[b, i, :], y[b, j, :])
        return o

    def test_sparseblockdot(self):
        # Compares the numpy version of sparseblockgemv to sparse_block_dot.

        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        o = sparse_block_dot(W, h, iIdx, b, oIdx)

        f = aesara.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()

        th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

        ref_out = self.gemv_numpy(
            b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val
        )

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemv(self):
        # Compares the numpy and aesara versions of sparseblockgemv.

        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        o = self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        f = aesara.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()

        th_out = f(W_val, h_val, iIdx_val, b_val, oIdx_val)
        ref_out = self.gemv_numpy(
            b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val
        )

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemvF(self):
        # Test the fortran order for W (which can happen in the grad for some
        # graphs).

        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        o = self.gemv_op(
            b.take(oIdx, axis=0),
            DimShuffle((False, False, False, False), (0, 1, 3, 2))(
                at.as_tensor_variable(W)
            ),
            h,
            iIdx,
            oIdx,
        )

        f = aesara.function([W, h, iIdx, b, oIdx], o, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()

        th_out = f(np.swapaxes(W_val, 2, 3), h_val, iIdx_val, b_val, oIdx_val)
        ref_out = self.gemv_numpy(
            b_val.take(oIdx_val, axis=0), W_val, h_val, iIdx_val, oIdx_val
        )

        utt.assert_allclose(ref_out, th_out)

    def test_sparseblockgemv_grad(self):
        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()

        iIdx = at.constant(iIdx_val)
        oIdx = at.constant(oIdx_val)

        def metaop(b, h, W):
            return sparse_block_dot(W, h, iIdx, b, oIdx)

        def op(b, h, W):
            return self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        eps = 3e-3
        utt.verify_grad(metaop, [b_val, h_val, W_val], mode=self.mode, eps=eps)
        utt.verify_grad(op, [b_val, h_val, W_val], mode=self.mode, eps=eps)

    def test_sparseblockgemv_grad_1(self):
        # Test that we correctly handle cases where dimensions are 1.
        rng = np.random.default_rng(230920)

        h_val = rng.standard_normal((1, 1, 1)).astype("float32")
        iIdx_val = rng.permutation(1)[:1][None, :]
        oIdx_val = rng.permutation(1)[:1][None, :]
        W_val = rng.standard_normal((1, 1, 1, 1)).astype("float32")
        b_val = rng.standard_normal((1, 1)).astype("float32")

        iIdx = at.constant(iIdx_val)
        oIdx = at.constant(oIdx_val)

        def metaop(b, h, W):
            return sparse_block_dot(W, h, iIdx, b, oIdx)

        def op(b, h, W):
            return self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)

        utt.verify_grad(metaop, [b_val, h_val, W_val], mode=self.mode)
        utt.verify_grad(op, [b_val, h_val, W_val], mode=self.mode)

    def test_sparseblockgemv_grad_shape(self):
        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        o = self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
        go = aesara.grad(o.sum(), [b, W, h])

        f = aesara.function([W, h, iIdx, b, oIdx], go, mode=self.mode)

        W_val, h_val, iIdx_val, b_val, oIdx_val = self.gemv_data()

        # just make sure that it runs correctly and all the shapes are ok.
        b_g, W_g, h_g = f(W_val, h_val, iIdx_val, b_val, oIdx_val)

        assert b_g.shape == b_val.shape
        assert h_g.shape == h_val.shape
        assert W_g.shape == W_val.shape

    def test_sparseblockouter(self):
        o = ftensor4()
        x = ftensor3()
        y = ftensor3()
        xIdx = imatrix()
        yIdx = imatrix()

        out = self.outer_op(o, x, y, xIdx, yIdx)

        f = aesara.function(
            [o, x, y, xIdx, yIdx], out, on_unused_input="warn", mode=self.mode
        )

        (
            o_val,
            x_val,
            y_val,
            xIdx_val,
            yIdx_val,
        ) = self.outer_data()

        th_out = f(o_val, x_val, y_val, xIdx_val, yIdx_val)
        ref_out = self.outer_numpy(o_val, x_val, y_val, xIdx_val, yIdx_val)

        utt.assert_allclose(ref_out, th_out)

    def test_dot_infershape(self):
        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        self._compile_and_check(
            [W, h, iIdx, b, oIdx],
            [sparse_block_dot(W, h, iIdx, b, oIdx)],
            self.gemv_data(),
            self.gemv_class,
        )

    def test_gemv_infershape(self):
        b = fmatrix()
        W = ftensor4()
        h = ftensor3()
        iIdx = imatrix()
        oIdx = imatrix()

        self._compile_and_check(
            [W, h, iIdx, b, oIdx],
            [self.gemv_op(b.take(oIdx, axis=0), W, h, iIdx, oIdx)],
            self.gemv_data(),
            self.gemv_class,
        )

    def test_outer_infershape(self):
        o = ftensor4()
        x = ftensor3()
        y = ftensor3()
        xIdx = imatrix()
        yIdx = imatrix()

        self._compile_and_check(
            [o, x, y, xIdx, yIdx],
            [self.outer_op(o, x, y, xIdx, yIdx)],
            self.outer_data(),
            self.outer_class,
        )

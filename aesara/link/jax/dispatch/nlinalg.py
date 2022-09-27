import jax.numpy as jnp

from aesara.link.jax.dispatch import jax_funcify
from aesara.tensor.blas import BatchedDot
from aesara.tensor.math import Dot, MaxAndArgmax
from aesara.tensor.nlinalg import SVD, Det, Eig, Eigh, MatrixInverse, QRFull


@jax_funcify.register(SVD)
def jax_funcify_SVD(op, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    def svd(x, full_matrices=full_matrices, compute_uv=compute_uv):
        return jnp.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

    return svd


@jax_funcify.register(Det)
def jax_funcify_Det(op, **kwargs):
    def det(x):
        return jnp.linalg.det(x)

    return det


@jax_funcify.register(Eig)
def jax_funcify_Eig(op, **kwargs):
    def eig(x):
        return jnp.linalg.eig(x)

    return eig


@jax_funcify.register(Eigh)
def jax_funcify_Eigh(op, **kwargs):
    uplo = op.UPLO

    def eigh(x, uplo=uplo):
        return jnp.linalg.eigh(x, UPLO=uplo)

    return eigh


@jax_funcify.register(MatrixInverse)
def jax_funcify_MatrixInverse(op, **kwargs):
    def matrix_inverse(x):
        return jnp.linalg.inv(x)

    return matrix_inverse


@jax_funcify.register(QRFull)
def jax_funcify_QRFull(op, **kwargs):
    mode = op.mode

    def qr_full(x, mode=mode):
        return jnp.linalg.qr(x, mode=mode)

    return qr_full


@jax_funcify.register(Dot)
def jax_funcify_Dot(op, **kwargs):
    def dot(x, y):
        return jnp.dot(x, y)

    return dot


@jax_funcify.register(BatchedDot)
def jax_funcify_BatchedDot(op, **kwargs):
    def batched_dot(a, b):
        if a.shape[0] != b.shape[0]:
            raise TypeError("Shapes must match in the 0-th dimension")
        if a.ndim == 2 or b.ndim == 2:
            return jnp.einsum("n...j,nj...->n...", a, b)
        return jnp.einsum("nij,njk->nik", a, b)

    return batched_dot


@jax_funcify.register(MaxAndArgmax)
def jax_funcify_MaxAndArgmax(op, **kwargs):
    axis = op.axis

    def maxandargmax(x, axis=axis):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        max_res = jnp.max(x, axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = jnp.array(
            [i for i in range(x.ndim) if i not in axes], dtype="int64"
        )
        # Not-reduced axes in front
        transposed_x = jnp.transpose(
            x, jnp.concatenate((keep_axes, jnp.array(axes, dtype="int64")))
        )
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = kept_shape + (
            jnp.prod(jnp.array(reduced_shape, dtype="int64"), dtype="int64"),
        )
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx_res = jnp.argmax(reshaped_x, axis=-1).astype("int64")

        return max_res, max_idx_res

    return maxandargmax

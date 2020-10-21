import numpy as np
import pytest

import theano
from theano import Apply, Op, config, tensor
from theano.gof import ParamsType
from theano.gpuarray.basic_ops import CGpuKernelBase
from theano.gpuarray.type import GpuArrayType, get_context, gpu_context_type
from theano.gradient import grad_undefined
from theano.scalar import int32 as int_t


# This is an implementation to test that CGpuKernelBase works and also
# to use as an example in the docs.  It is not used for user graphs.
class GpuEye(CGpuKernelBase, Op):
    """
    Eye for GPU.

    """

    __props__ = ("dtype", "context_name")
    params_type = ParamsType(typecode=int_t, context=gpu_context_type)

    def __init__(self, dtype=None, context_name=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype
        self.context_name = context_name
        CGpuKernelBase.__init__(
            self, ["c_code/tstgpueye.c"], "APPLY_SPECIFIC(tstgpueye)"
        )

    def get_params(self, node):
        pygpu_gpuarray = pytest.importorskip("pygpu.gpuarray")

        return self.params_type.get_params(
            typecode=pygpu_gpuarray.dtype_to_typecode(self.dtype),
            context=get_context(self.context_name),
        )

    def c_headers(self):
        return ["<gpuarray/types.h>", "<gpuarray/kernel.h>"]

    def make_node(self, n, m):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        assert n.ndim == 0
        assert m.ndim == 0
        otype = GpuArrayType(
            dtype=self.dtype,
            broadcastable=(False, False),
            context_name=self.context_name,
        )

        return Apply(self, [n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in range(2)]


def test_cgpukernelbase():
    # Import inside the function to prevent the back-end from being
    # initialized when reloading the GpuEye object from cache.
    from .config import mode_with_gpu, test_ctx_name

    op = GpuEye(dtype="int32", context_name=test_ctx_name)

    f = theano.function([], op(4, 5), mode=mode_with_gpu)

    r = f()

    assert r.dtype == "int32"
    assert (np.asarray(r) == np.eye(4, 5, dtype="int32")).all()

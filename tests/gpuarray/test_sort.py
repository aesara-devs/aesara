from theano.gpuarray.sort import GpuTopKOp

from tests.tensor.test_sort import TestTopK
from tests.gpuarray.config import mode_with_gpu


class TestGpuTopK(TestTopK):
    mode = mode_with_gpu
    dtype = "float32"
    op_class = GpuTopKOp

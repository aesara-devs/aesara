from aesara.gpuarray.sort import GpuTopKOp
from tests.gpuarray.config import mode_with_gpu
from tests.tensor.test_sort import TestTopK


class TestGpuTopK(TestTopK):
    mode = mode_with_gpu
    dtype = "float32"
    op_class = GpuTopKOp

import theano
import theano.tensor.tests.test_sort
from .config import mode_with_gpu
from ..sort import GpuTopKOp


class TestGpuTopK(theano.tensor.tests.test_sort.TestTopK):
    mode = mode_with_gpu
    dtype = "float32"
    op_class = GpuTopKOp

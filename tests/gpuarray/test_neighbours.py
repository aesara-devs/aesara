from aesara.gpuarray.neighbours import GpuImages2Neibs
from tests.gpuarray.config import mode_with_gpu
from tests.tensor.nnet import test_neighbours


class TestGpuImages2Neibs(test_neighbours.TestImages2Neibs):
    mode = mode_with_gpu
    op = GpuImages2Neibs
    dtypes = ["int64", "float32", "float64"]

from tests.gpuarray.config import mode_with_gpu
from tests.tensor.nnet import test_neighbours
from theano.gpuarray.neighbours import GpuImages2Neibs


class TestGpuImages2Neibs(test_neighbours.TestImages2Neibs):
    mode = mode_with_gpu
    op = GpuImages2Neibs
    dtypes = ["int64", "float32", "float64"]

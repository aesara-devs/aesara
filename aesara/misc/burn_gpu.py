"""This script trigger convolution operation. We think it cause more
GPU power consumption then gemm call.

"""


import numpy as np

import aesara
from aesara.configdefaults import config
from aesara.gpuarray import dnn
from aesara.tensor.nnet.abstract_conv import get_conv_output_shape
from aesara.tensor.type import tensor4


def burn():
    sz = 128
    img_shp = [sz, sz, sz, sz]
    kern_shp = [sz // 2, sz, 3, 3]
    out_shp = get_conv_output_shape(img_shp, kern_shp, "valid", (1, 1))
    img = tensor4("img")
    kern = tensor4("kern")
    out = tensor4("out")

    def rand(shp):
        return np.random.rand(*shp).astype(config.floatX)

    img = aesara.shared(rand(img_shp))
    kern = aesara.shared(rand(kern_shp))
    out = aesara.shared(rand(out_shp))
    # beta 1 is needed to force the reuse of out, otherwise, it is
    # replaced by a GpuAllocEmpty
    o1 = dnn._dnn_conv(img, kern, conv_mode="conv", out=out, beta=1.0)
    mode = aesara.compile.get_default_mode().including("local_remove_all_assert")
    f = aesara.function([], [o1], mode=mode)
    aesara.printing.debugprint(f)
    print("Start computation")
    for i in range(10000):
        f.fn()
    print("Computation stopped")


if __name__ == "__main__":
    burn()

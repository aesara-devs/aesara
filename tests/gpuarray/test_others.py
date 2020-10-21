import numpy as np
import pytest


pygpu = pytest.importorskip("pygpu")

from tests.gpuarray.config import mode_with_gpu, test_ctx_name
from tests.misc.test_may_share_memory import may_share_memory_core
from tests.tensor import test_opt
from theano.gpuarray.basic_ops import GpuFromHost, HostFromGpu
from theano.gpuarray.type import (
    GpuArraySharedVariable,
    GpuArrayType,
    get_context,
    gpuarray_shared_constructor,
)
from theano.misc.pkl_utils import dump, load


class TestFusion(test_opt.TestFusion):
    mode = mode_with_gpu.excluding("local_dnn_reduction")
    _shared = staticmethod(gpuarray_shared_constructor)
    topo_exclude = (GpuFromHost, HostFromGpu)


def test_may_share_memory():
    ctx = get_context(test_ctx_name)
    a = pygpu.empty((5, 4), context=ctx)
    b = pygpu.empty((5, 4), context=ctx)

    may_share_memory_core(a, b)


def test_dump_load():
    x = GpuArraySharedVariable(
        "x",
        GpuArrayType("float32", (1, 1), name="x", context_name=test_ctx_name),
        [[1]],
        False,
    )

    with open("test", "wb") as f:
        dump(x, f)

    with open("test", "rb") as f:
        x = load(f)

    assert x.name == "x"
    np.testing.assert_allclose(x.get_value(), [[1]])

import numpy as np
import pytest
import scipy as sp

import aesara
from aesara import sparse
from aesara.compile.mode import Mode, get_default_mode
from aesara.configdefaults import config
from aesara.sparse.rewriting import SamplingDotCSR, sd_csc
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.math import sum as at_sum
from aesara.tensor.type import ivector, matrix, vector
from tests import unittest_tools as utt
from tests.sparse.test_basic import random_lil


def test_local_csm_properties_csm():
    data = vector()
    indices, indptr, shape = (ivector(), ivector(), ivector())
    mode = get_default_mode()
    mode = mode.including("specialize", "local_csm_properties_csm")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        f = aesara.function(
            [data, indices, indptr, shape],
            sparse.csm_properties(CS(data, indices, indptr, shape)),
            mode=mode,
        )
        assert not any(
            isinstance(node.op, (sparse.CSM, sparse.CSMProperties))
            for node in f.maker.fgraph.toposort()
        )
        v = cast(random_lil((10, 40), config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


@pytest.mark.skip(reason="Rewrite disabled as it don't support unsorted indices")
@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_csm_grad_c():
    data = vector()
    indices, indptr, shape = (ivector(), ivector(), ivector())
    mode = get_default_mode()

    if aesara.config.mode == "FAST_COMPILE":
        mode = Mode(linker="c|py", optimizer="fast_compile")

    mode = mode.including("specialize", "local_csm_grad_c")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        cost = at_sum(sparse.DenseFromSparse()(CS(data, indices, indptr, shape)))
        f = aesara.function(
            [data, indices, indptr, shape], aesara.grad(cost, data), mode=mode
        )
        assert not any(
            isinstance(node.op, sparse.CSMGrad) for node in f.maker.fgraph.toposort()
        )
        v = cast(random_lil((10, 40), config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_d():
    mode = get_default_mode()
    mode = mode.including("specialize", "local_mul_s_d")

    for sp_format in sparse.sparse_formats:
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), matrix()]

        f = aesara.function(inputs, sparse.mul_s_d(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSD) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_v():
    mode = get_default_mode()
    mode = mode.including("specialize", "local_mul_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), vector()]

        f = aesara.function(inputs, sparse.mul_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSV) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_structured_add_s_v():
    mode = get_default_mode()
    mode = mode.including("specialize", "local_structured_add_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(aesara.sparse, sp_format + "_matrix")(), vector()]

        f = aesara.function(inputs, sparse.structured_add_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.StructuredAddSV)
            for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_sampling_dot_csr():
    mode = get_default_mode()
    mode = mode.including("specialize", "local_sampling_dot_csr")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [
            matrix(),
            matrix(),
            getattr(aesara.sparse, sp_format + "_matrix")(),
        ]

        f = aesara.function(inputs, sparse.sampling_dot(*inputs), mode=mode)

        if aesara.config.blas__ldflags:
            assert not any(
                isinstance(node.op, sparse.SamplingDot)
                for node in f.maker.fgraph.toposort()
            )
        else:
            # SamplingDotCSR's C implementation needs blas, so it should not
            # be inserted
            assert not any(
                isinstance(node.op, SamplingDotCSR)
                for node in f.maker.fgraph.toposort()
            )


def test_local_dense_from_sparse_sparse_from_dense():
    mode = get_default_mode()
    mode = mode.including("local_dense_from_sparse_sparse_from_dense")

    m = matrix()
    for op in [aesara.sparse.csr_from_dense, aesara.sparse.csc_from_dense]:
        s = op(m)
        o = aesara.sparse.dense_from_sparse(s)
        f = aesara.function([m], o, mode=mode)
        # We should just have a deep copy.
        assert len(f.maker.fgraph.apply_nodes) == 1
        f([[1, 2], [3, 4]])


def test_sd_csc():
    A = sp.sparse.random(4, 5, density=0.60, format="csc", dtype=np.float32)
    b = np.random.random((5, 2)).astype(np.float32)
    target = A * b

    a_val = as_tensor_variable(A.data)
    a_ind = as_tensor_variable(A.indices)
    a_ptr = as_tensor_variable(A.indptr)
    nrows = as_tensor_variable(np.int32(A.shape[0]))
    b = as_tensor_variable(b)

    res = sd_csc(a_val, a_ind, a_ptr, nrows, b).eval()

    utt.assert_allclose(res, target)

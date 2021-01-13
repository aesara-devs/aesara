import pytest


sp = pytest.importorskip("scipy", minversion="0.7.0")

import numpy as np

import theano
import theano.tensor as tt
from tests import unittest_tools as utt
from tests.sparse.test_basic import random_lil
from theano import sparse
from theano.configdefaults import config
from theano.tensor.type import ivector, matrix, vector


def test_local_csm_properties_csm():
    data = vector()
    indices, indptr, shape = (ivector(), ivector(), ivector())
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_csm_properties_csm")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        f = theano.function(
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


@pytest.mark.skip(reason="Opt disabled as it don't support unsorted indices")
@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_csm_grad_c():
    data = vector()
    indices, indptr, shape = (ivector(), ivector(), ivector())
    mode = theano.compile.mode.get_default_mode()

    if theano.config.mode == "FAST_COMPILE":
        mode = theano.compile.Mode(linker="c|py", optimizer="fast_compile")

    mode = mode.including("specialize", "local_csm_grad_c")
    for CS, cast in [
        (sparse.CSC, sp.sparse.csc_matrix),
        (sparse.CSR, sp.sparse.csr_matrix),
    ]:
        cost = tt.sum(sparse.DenseFromSparse()(CS(data, indices, indptr, shape)))
        f = theano.function(
            [data, indices, indptr, shape], theano.grad(cost, data), mode=mode
        )
        assert not any(
            isinstance(node.op, sparse.CSMGrad) for node in f.maker.fgraph.toposort()
        )
        v = cast(random_lil((10, 40), config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_d():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_d")

    for sp_format in sparse.sparse_formats:
        inputs = [getattr(theano.sparse, sp_format + "_matrix")(), matrix()]

        f = theano.function(inputs, sparse.mul_s_d(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSD) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_mul_s_v():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + "_matrix")(), vector()]

        f = theano.function(inputs, sparse.mul_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.MulSV) for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_structured_add_s_v():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_structured_add_s_v")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + "_matrix")(), vector()]

        f = theano.function(inputs, sparse.structured_add_s_v(*inputs), mode=mode)

        assert not any(
            isinstance(node.op, sparse.StructuredAddSV)
            for node in f.maker.fgraph.toposort()
        )


@pytest.mark.skipif(
    not theano.config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_local_sampling_dot_csr():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_sampling_dot_csr")

    for sp_format in ["csr"]:  # Not implemented for other format
        inputs = [
            matrix(),
            matrix(),
            getattr(theano.sparse, sp_format + "_matrix")(),
        ]

        f = theano.function(inputs, sparse.sampling_dot(*inputs), mode=mode)

        if theano.config.blas__ldflags:
            assert not any(
                isinstance(node.op, sparse.SamplingDot)
                for node in f.maker.fgraph.toposort()
            )
        else:
            # SamplingDotCSR's C implementation needs blas, so it should not
            # be inserted
            assert not any(
                isinstance(node.op, sparse.opt.SamplingDotCSR)
                for node in f.maker.fgraph.toposort()
            )


def test_local_dense_from_sparse_sparse_from_dense():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("local_dense_from_sparse_sparse_from_dense")

    m = matrix()
    for op in [theano.sparse.csr_from_dense, theano.sparse.csc_from_dense]:
        s = op(m)
        o = theano.sparse.dense_from_sparse(s)
        f = theano.function([m], o, mode=mode)
        # We should just have a deep copy.
        assert len(f.maker.fgraph.apply_nodes) == 1
        f([[1, 2], [3, 4]])


def test_sd_csc():

    A = sp.sparse.rand(4, 5, density=0.60, format="csc", dtype=np.float32)
    b = np.random.rand(5, 2).astype(np.float32)
    target = A * b

    a_val = tt.as_tensor_variable(A.data)
    a_ind = tt.as_tensor_variable(A.indices)
    a_ptr = tt.as_tensor_variable(A.indptr)
    nrows = tt.as_tensor_variable(np.int32(A.shape[0]))
    b = tt.as_tensor_variable(b)

    res = theano.sparse.opt.sd_csc(a_val, a_ind, a_ptr, nrows, b).eval()

    utt.assert_allclose(res, target)

import aesara
from aesara.graph.rewriting.basic import check_stack_trace
from aesara.tensor.nnet.blocksparse import (
    sparse_block_dot,
    sparse_block_gemv,
    sparse_block_gemv_inplace,
    sparse_block_outer,
    sparse_block_outer_inplace,
)
from aesara.tensor.type import fmatrix, ftensor3, ftensor4, lmatrix
from tests.unittest_tools import assertFailure_fast


def test_blocksparse_inplace_gemv_opt():
    b = fmatrix()
    W = ftensor4()
    h = ftensor3()
    iIdx = lmatrix()
    oIdx = lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = aesara.function([W, h, iIdx, b, oIdx], o)

    if aesara.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=[sparse_block_gemv])
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=[sparse_block_gemv_inplace])


if aesara.config.mode != "FAST_COMPILE":
    test_blocksparse_inplace_gemv_opt = assertFailure_fast(
        test_blocksparse_inplace_gemv_opt
    )


def test_blocksparse_inplace_outer_opt():
    b = fmatrix()
    W = ftensor4()
    h = ftensor3()
    iIdx = lmatrix()
    oIdx = lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = aesara.function(
        [W, h, iIdx, b, oIdx], [o, aesara.gradient.grad(o.sum(), wrt=W)]
    )

    if aesara.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=sparse_block_outer)
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=sparse_block_outer_inplace)

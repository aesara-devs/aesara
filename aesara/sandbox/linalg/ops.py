import logging

from aesara.graph.rewriting.basic import node_rewriter
from aesara.tensor import basic as at
from aesara.tensor.blas import Dot22
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.math import Dot, Prod, dot, log
from aesara.tensor.math import pow as at_pow
from aesara.tensor.math import prod
from aesara.tensor.nlinalg import Det, MatrixInverse, trace
from aesara.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from aesara.tensor.slinalg import Cholesky, Solve, cholesky, solve


logger = logging.getLogger(__name__)


@register_canonicalize
@node_rewriter([DimShuffle])
def transinv_to_invtrans(fgraph, node):
    if isinstance(node.op, DimShuffle):
        if node.op.new_order == (1, 0):
            (A,) = node.inputs
            if A.owner:
                if isinstance(A.owner.op, MatrixInverse):
                    (X,) = A.owner.inputs
                    return [A.owner.op(node.op(X))]


@register_stabilize
@node_rewriter([Dot, Dot22])
def inv_as_solve(fgraph, node):
    """
    This utilizes a boolean `symmetric` tag on the matrices.
    """
    if isinstance(node.op, (Dot, Dot22)):
        l, r = node.inputs
        if l.owner and isinstance(l.owner.op, MatrixInverse):
            return [solve(l.owner.inputs[0], r)]
        if r.owner and isinstance(r.owner.op, MatrixInverse):
            x = r.owner.inputs[0]
            if getattr(x.tag, "symmetric", None) is True:
                return [solve(x, l.T).T]
            else:
                return [solve(x.T, l.T).T]


@register_stabilize
@register_canonicalize
@node_rewriter([Solve])
def tag_solve_triangular(fgraph, node):
    """
    If a general solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.

    """
    if isinstance(node.op, Solve):
        if node.op.assume_a == "gen":
            A, b = node.inputs  # result is solution Ax=b
            if A.owner and isinstance(A.owner.op, Cholesky):
                if A.owner.op.lower:
                    return [Solve(assume_a="sym", lower=True)(A, b)]
                else:
                    return [Solve(assume_a="sym", lower=False)(A, b)]
            if (
                A.owner
                and isinstance(A.owner.op, DimShuffle)
                and A.owner.op.new_order == (1, 0)
            ):
                (A_T,) = A.owner.inputs
                if A_T.owner and isinstance(A_T.owner.op, Cholesky):
                    if A_T.owner.op.lower:
                        return [Solve(assume_a="sym", lower=False)(A, b)]
                    else:
                        return [Solve(assume_a="sym", lower=True)(A, b)]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([DimShuffle])
def no_transpose_symmetric(fgraph, node):
    if isinstance(node.op, DimShuffle):
        x = node.inputs[0]
        if x.type.ndim == 2 and getattr(x.tag, "symmetric", None) is True:
            if node.op.new_order == [1, 0]:
                return [x]


@register_stabilize
@node_rewriter([Solve])
def psd_solve_with_chol(fgraph, node):
    """
    This utilizes a boolean `psd` tag on matrices.
    """
    if isinstance(node.op, Solve):
        A, b = node.inputs  # result is solution Ax=b
        if getattr(A.tag, "psd", None) is True:
            L = cholesky(A)
            # N.B. this can be further reduced to a yet-unwritten cho_solve Op
            #     __if__ no other Op makes use of the the L matrix during the
            #     stabilization
            Li_b = Solve(assume_a="sym", lower=True)(L, b)
            x = Solve(assume_a="sym", lower=False)(L.T, Li_b)
            return [x]


@register_stabilize
@register_specialize
@node_rewriter([Det])
def local_det_chol(fgraph, node):
    """
    If we have det(X) and there is already an L=cholesky(X)
    floating around, then we can use prod(diag(L)) to get the determinant.

    """
    if isinstance(node.op, Det):
        (x,) = node.inputs
        for (cl, xpos) in fgraph.clients[x]:
            if isinstance(cl.op, Cholesky):
                L = cl.outputs[0]
                return [prod(at.extract_diag(L) ** 2)]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_prod_sqr(fgraph, node):
    """
    This utilizes a boolean `positive` tag on matrices.
    """
    if node.op == log:
        (x,) = node.inputs
        if x.owner and isinstance(x.owner.op, Prod):
            # we cannot always make this substitution because
            # the prod might include negative terms
            p = x.owner.inputs[0]

            # p is the matrix we're reducing with prod
            if getattr(p.tag, "positive", None) is True:
                return [log(p).sum(axis=x.owner.op.axis)]

            # TODO: have a reduction like prod and sum that simply
            # returns the sign of the prod multiplication.


def spectral_radius_bound(X, log2_exponent):
    """
    Returns upper bound on the largest eigenvalue of square symmetric matrix X.

    log2_exponent must be a positive-valued integer. The larger it is, the
    slower and tighter the bound. Values up to 5 should usually suffice. The
    algorithm works by multiplying X by itself this many times.

    From V.Pan, 1990. "Estimating the Extremal Eigenvalues of a Symmetric
    Matrix", Computers Math Applic. Vol 20 n. 2 pp 17-22.
    Rq: an efficient algorithm, not used here, is defined in this paper.

    """
    if X.type.ndim != 2:
        raise TypeError("spectral_radius_bound requires a matrix argument", X)
    if not isinstance(log2_exponent, int):
        raise TypeError(
            "spectral_radius_bound requires an integer exponent", log2_exponent
        )
    if log2_exponent <= 0:
        raise ValueError(
            "spectral_radius_bound requires a strictly positive " "exponent",
            log2_exponent,
        )

    XX = X
    for i in range(log2_exponent):
        XX = dot(XX, XX)
    return at_pow(trace(XX), 2 ** (-log2_exponent))

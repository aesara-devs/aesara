import jax

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.slinalg import Cholesky, Solve, SolveTriangular


@jax_funcify.register(Cholesky)
def jax_funcify_Cholesky(op, **kwargs):
    lower = op.lower

    def cholesky(a, lower=lower):
        return jax.scipy.linalg.cholesky(a, lower=lower).astype(a.dtype)

    return cholesky


@jax_funcify.register(Solve)
def jax_funcify_Solve(op, **kwargs):

    if op.assume_a != "gen" and op.lower:
        lower = True
    else:
        lower = False

    def solve(a, b, lower=lower):
        return jax.scipy.linalg.solve(a, b, lower=lower)

    return solve


@jax_funcify.register(SolveTriangular)
def jax_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower
    trans = op.trans
    unit_diagonal = op.unit_diagonal
    check_finite = op.check_finite

    def solve_triangular(A, b):
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            lower=lower,
            trans=trans,
            unit_diagonal=unit_diagonal,
            check_finite=check_finite,
        )

    return solve_triangular

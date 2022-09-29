import logging
import warnings
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.linalg
from typing_extensions import Literal

import aesara.tensor
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor import as_tensor_variable
from aesara.tensor import basic as at
from aesara.tensor import math as atm
from aesara.tensor.shape import reshape
from aesara.tensor.type import matrix, tensor, vector
from aesara.tensor.var import TensorVariable


if TYPE_CHECKING:
    from aesara.tensor import TensorLike


logger = logging.getLogger(__name__)


class Cholesky(Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.

    Parameters
    ----------
    lower : bool, default=True
        Whether to return the lower or upper cholesky factor
    on_error : ['raise', 'nan']
        If on_error is set to 'raise', this Op will raise a
        `scipy.linalg.LinAlgError` if the matrix is not positive definite.
        If on_error is set to 'nan', it will return a matrix containing
        nans instead.
    """

    # TODO: inplace
    # TODO: for specific dtypes
    # TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ("lower", "destructive", "on_error")

    def __init__(self, lower=True, on_error="raise"):
        self.lower = lower
        self.destructive = False
        if on_error not in ("raise", "nan"):
            raise ValueError('on_error must be one of "raise" or ""nan"')
        self.on_error = on_error

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)
        except scipy.linalg.LinAlgError:
            if self.on_error == "raise":
                raise
            else:
                z[0] = (np.zeros(x.shape) * np.nan).astype(x.dtype)

    def L_op(self, inputs, outputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [#]_

        References
        ----------
        .. [#] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """

        dz = gradients[0]
        chol_x = outputs[0]

        # Replace the cholesky decomposition with 1 if there are nans
        # or solve_upper_triangular will throw a ValueError.
        if self.on_error == "nan":
            ok = ~atm.any(atm.isnan(chol_x))
            chol_x = at.switch(ok, chol_x, 1)
            dz = at.switch(ok, dz, 1)

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return at.tril(mtx) - at.diag(at.diagonal(mtx) / 2.0)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            solve_upper = SolveTriangular(lower=False)
            return solve_upper(outer.T, solve_upper(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz))
        )

        if self.lower:
            grad = at.tril(s + s.T) - at.diag(at.diagonal(s))
        else:
            grad = at.triu(s + s.T) - at.diag(at.diagonal(s))

        if self.on_error == "nan":
            return [at.switch(ok, grad, np.nan)]
        else:
            return [grad]


cholesky = Cholesky()


class CholeskySolve(Op):

    __props__ = ("lower", "check_finite")

    def __init__(
        self,
        lower=True,
        check_finite=True,
    ):
        self.lower = lower
        self.check_finite = check_finite

    def __repr__(self):
        return "CholeskySolve{%s}" % str(self._props())

    def make_node(self, C, b):
        C = as_tensor_variable(C)
        b = as_tensor_variable(b)
        assert C.ndim == 2
        assert b.ndim in (1, 2)

        # infer dtype by solving the most simple
        # case with (1, 1) matrices
        o_dtype = scipy.linalg.solve(
            np.eye(1).astype(C.dtype), np.eye(1).astype(b.dtype)
        ).dtype
        x = tensor(dtype=o_dtype, shape=b.type.shape)
        return Apply(self, [C, b], [x])

    def perform(self, node, inputs, output_storage):
        C, b = inputs
        rval = scipy.linalg.cho_solve(
            (C, self.lower),
            b,
            check_finite=self.check_finite,
        )

        output_storage[0][0] = rval

    def infer_shape(self, fgraph, node, shapes):
        Cshape, Bshape = shapes
        rows = Cshape[1]
        if len(Bshape) == 1:  # b is a Vector
            return [(rows,)]
        else:
            cols = Bshape[1]  # b is a Matrix
            return [(rows, cols)]


cho_solve = CholeskySolve()


def cho_solve(c_and_lower, b, check_finite=True):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    """

    A, lower = c_and_lower
    return CholeskySolve(lower=lower, check_finite=check_finite)(A, b)


class SolveBase(Op):
    """Base class for `scipy.linalg` matrix equation solvers."""

    __props__ = (
        "lower",
        "check_finite",
    )

    def __init__(
        self,
        lower=False,
        check_finite=True,
    ):
        self.lower = lower
        self.check_finite = check_finite

    def perform(self, node, inputs, outputs):
        pass

    def make_node(self, A, b):
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)

        if A.ndim != 2:
            raise ValueError(f"`A` must be a matrix; got {A.type} instead.")
        if b.ndim not in (1, 2):
            raise ValueError(f"`b` must be a matrix or a vector; got {b.type} instead.")

        # Infer dtype by solving the most simple case with 1x1 matrices
        o_dtype = scipy.linalg.solve(
            np.eye(1).astype(A.dtype), np.eye(1).astype(b.dtype)
        ).dtype
        x = tensor(dtype=o_dtype, shape=b.type.shape)
        return Apply(self, [A, b], [x])

    def infer_shape(self, fgraph, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:
            return [(rows,)]
        else:
            cols = Bshape[1]
            return [(rows, cols)]

    def L_op(self, inputs, outputs, output_gradients):
        r"""Reverse-mode gradient updates for matrix solve operation :math:`c = A^{-1} b`.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, b = inputs

        c = outputs[0]
        # C is a scalar representing the entire graph
        # `output_gradients` is (dC/dc,)
        # We need to return (dC/d[inv(A)], dC/db)
        c_bar = output_gradients[0]

        trans_solve_op = type(self)(
            **{
                k: (not getattr(self, k) if k == "lower" else getattr(self, k))
                for k in self.__props__
            }
        )
        b_bar = trans_solve_op(A.T, c_bar)
        # force outer product if vector second input
        A_bar = -atm.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)

        return [A_bar, b_bar]

    def __repr__(self):
        return f"{type(self).__name__}{self._props()}"


class SolveTriangular(SolveBase):
    """Solve a system of linear equations."""

    __props__ = (
        "lower",
        "trans",
        "unit_diagonal",
        "check_finite",
    )

    def __init__(
        self,
        trans=0,
        lower=False,
        unit_diagonal=False,
        check_finite=True,
    ):
        super().__init__(lower=lower, check_finite=check_finite)
        self.trans = trans
        self.unit_diagonal = unit_diagonal

    def perform(self, node, inputs, outputs):
        A, b = inputs
        outputs[0][0] = scipy.linalg.solve_triangular(
            A,
            b,
            lower=self.lower,
            trans=self.trans,
            unit_diagonal=self.unit_diagonal,
            check_finite=self.check_finite,
        )

    def L_op(self, inputs, outputs, output_gradients):
        res = super().L_op(inputs, outputs, output_gradients)

        if self.lower:
            res[0] = at.tril(res[0])
        else:
            res[0] = at.triu(res[0])

        return res


solvetriangular = SolveTriangular()


def solve_triangular(
    a: TensorVariable,
    b: TensorVariable,
    trans: Union[int, str] = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    check_finite: bool = True,
) -> TensorVariable:
    """Solve the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Parameters
    ----------
    a
        Square input data
    b
        Input data for the right hand side.
    lower : bool, optional
        Use only data contained in the lower triangle of `a`. Default is to use upper triangle.
    trans: {0, 1, 2, ‘N’, ‘T’, ‘C’}, optional
        Type of system to solve:
        trans       system
        0 or 'N'    a x = b
        1 or 'T'    a^T x = b
        2 or 'C'    a^H x = b
    unit_diagonal: bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and will not be referenced.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    """
    return SolveTriangular(
        lower=lower,
        trans=trans,
        unit_diagonal=unit_diagonal,
        check_finite=check_finite,
    )(a, b)


class Solve(SolveBase):
    """
    Solve a system of linear equations.
    """

    __props__ = (
        "assume_a",
        "lower",
        "check_finite",
    )

    def __init__(
        self,
        assume_a="gen",
        lower=False,
        check_finite=True,
    ):
        if assume_a not in ("gen", "sym", "her", "pos"):
            raise ValueError(f"{assume_a} is not a recognized matrix structure")

        super().__init__(lower=lower, check_finite=check_finite)
        self.assume_a = assume_a

    def perform(self, node, inputs, outputs):
        a, b = inputs
        outputs[0][0] = scipy.linalg.solve(
            a=a,
            b=b,
            lower=self.lower,
            check_finite=self.check_finite,
            assume_a=self.assume_a,
        )


solve = Solve()


def solve(a, b, assume_a="gen", lower=False, check_finite=True):
    """Solves the linear equation set ``a * x = b`` for the unknown ``x`` for square ``a`` matrix.

    If the data matrix is known to be a particular type then supplying the
    corresponding string to ``assume_a`` key chooses the dedicated solver.
    The available options are

    ===================  ========
    generic matrix       'gen'
    symmetric            'sym'
    hermitian            'her'
    positive definite    'pos'
    ===================  ========

    If omitted, ``'gen'`` is the default structure.

    The datatype of the arrays define which solver is called regardless
    of the values. In other words, even when the complex array entries have
    precisely zero imaginary parts, the complex solver will be called based
    on the data type of the array.

    Parameters
    ----------
    a : (N, N) array_like
        Square input data
    b : (N, NRHS) array_like
        Input data for the right hand side.
    lower : bool, optional
        If True, only the data contained in the lower triangle of `a`. Default
        is to use upper triangle. (ignored for ``'gen'``)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    assume_a : str, optional
        Valid entries are explained above.
    """
    return Solve(
        lower=lower,
        check_finite=check_finite,
        assume_a=assume_a,
    )(a, b)


class Eigvalsh(Op):
    """
    Generalized eigenvalues of a Hermitian positive definite eigensystem.

    """

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b):
        if b == aesara.tensor.type_other.NoneConst:
            a = as_tensor_variable(a)
            assert a.ndim == 2

            out_dtype = aesara.scalar.upcast(a.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a], [w])
        else:
            a = as_tensor_variable(a)
            b = as_tensor_variable(b)
            assert a.ndim == 2
            assert b.ndim == 2

            out_dtype = aesara.scalar.upcast(a.dtype, b.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a, b], [w])

    def perform(self, node, inputs, outputs):
        (w,) = outputs
        if len(inputs) == 2:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=inputs[1], lower=self.lower)
        else:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=None, lower=self.lower)

    def grad(self, inputs, g_outputs):
        a, b = inputs
        (gw,) = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """
    Gradient of generalized eigenvalues of a Hermitian positive definite
    eigensystem.

    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of aesara ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on GitHub at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = np.tril
            self.tri1 = lambda a: np.triu(a, 1)
        else:
            self.tri0 = np.triu
            self.tri1 = lambda a: np.tril(a, -1)

    def make_node(self, a, b, gw):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        gw = as_tensor_variable(gw)
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = aesara.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = matrix(dtype=out_dtype)
        out2 = matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, inputs, outputs):
        (a, b, gw) = inputs
        w, v = scipy.linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(np.diag(gw).dot(v.T))
        gB = -v.dot(np.diag(gw * w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = np.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)


def kron(a, b):
    """Kronecker product.

    Same as scipy.linalg.kron(a, b).

    Parameters
    ----------
    a: array_like
    b: array_like

    Returns
    -------
    array_like with a.ndim + b.ndim - 2 dimensions

    Notes
    -----
    numpy.kron(a, b) != scipy.linalg.kron(a, b)!
    They don't have the same shape and order when
    a.ndim != b.ndim != 2.

    """
    a = as_tensor_variable(a)
    b = as_tensor_variable(b)
    if a.ndim + b.ndim <= 2:
        raise TypeError(
            "kron: inputs dimensions must sum to 3 or more. "
            f"You passed {int(a.ndim)} and {int(b.ndim)}."
        )
    o = atm.outer(a, b)
    o = o.reshape(at.concatenate((a.shape, b.shape)), a.ndim + b.ndim)
    shf = o.dimshuffle(0, 2, 1, *list(range(3, o.ndim)))
    if shf.ndim == 3:
        shf = o.dimshuffle(1, 0, 2)
        o = shf.flatten()
    else:
        o = shf.reshape(
            (o.shape[0] * o.shape[2], o.shape[1] * o.shape[3])
            + tuple(o.shape[i] for i in range(4, o.ndim))
        )
    return o


class Expm(Op):
    """
    Compute the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2
        expm = matrix(dtype=A.dtype)
        return Apply(
            self,
            [
                A,
            ],
            [
                expm,
            ],
        )

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy.linalg.expm(A)

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [ExpmGrad()(A, g_out)]

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


class ExpmGrad(Op):
    """
    Gradient of the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A, gw):
        A = as_tensor_variable(A)
        assert A.ndim == 2
        out = matrix(dtype=A.dtype)
        return Apply(
            self,
            [A, gw],
            [
                out,
            ],
        )

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A, gA) = inputs
        (out,) = outputs
        w, V = scipy.linalg.eig(A, right=True)
        U = scipy.linalg.inv(V).T

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            out[0] = Y.astype(A.dtype)


expm = Expm()


class SolveContinuousLyapunov(Op):
    __props__ = ()

    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)

        out_dtype = aesara.scalar.upcast(A.dtype, B.dtype)
        X = aesara.tensor.matrix(dtype=out_dtype)

        return aesara.graph.basic.Apply(self, [A, B], [X])

    def perform(self, node, inputs, output_storage):
        (A, B) = inputs
        X = output_storage[0]

        X[0] = scipy.linalg.solve_continuous_lyapunov(A, B)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        # Note that they write the equation as AX + XA.H + Q = 0, while scipy uses AX + XA^H = Q,
        # so minor adjustments need to be made.
        A, Q = inputs
        (dX,) = output_grads

        X = self(A, Q)
        S = self(A.conj().T, -dX)  # Eq 31, adjusted

        A_bar = S.dot(X.conj().T) + S.conj().T.dot(X)
        Q_bar = -S  # Eq 29, adjusted

        return [A_bar, Q_bar]


class BilinearSolveDiscreteLyapunov(Op):
    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)

        out_dtype = aesara.scalar.upcast(A.dtype, B.dtype)
        X = aesara.tensor.matrix(dtype=out_dtype)

        return aesara.graph.basic.Apply(self, [A, B], [X])

    def perform(self, node, inputs, output_storage):
        (A, B) = inputs
        X = output_storage[0]

        X[0] = scipy.linalg.solve_discrete_lyapunov(A, B, method="bilinear")

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, Q = inputs
        (dX,) = output_grads

        X = self(A, Q)

        # Eq 41, note that it is not written as a proper Lyapunov equation
        S = self(A.conj().T, dX)

        A_bar = aesara.tensor.linalg.matrix_dot(
            S, A, X.conj().T
        ) + aesara.tensor.linalg.matrix_dot(S.conj().T, A, X)
        Q_bar = S
        return [A_bar, Q_bar]


_solve_continuous_lyapunov = SolveContinuousLyapunov()
_solve_bilinear_direct_lyapunov = BilinearSolveDiscreteLyapunov()


def iscomplexobj(x):
    type_ = x.type
    dtype = type_.dtype
    return "complex" in dtype


def _direct_solve_discrete_lyapunov(A: "TensorLike", Q: "TensorLike") -> TensorVariable:
    A_ = as_tensor_variable(A)
    Q_ = as_tensor_variable(Q)

    if "complex" in A_.type.dtype:
        AA = kron(A_, A_.conj())
    else:
        AA = kron(A_, A_)

    X = solve(at.eye(AA.shape[0]) - AA, Q_.ravel())
    return reshape(X, Q_.shape)


def solve_discrete_lyapunov(
    A: "TensorLike", Q: "TensorLike", method: Literal["direct", "bilinear"] = "direct"
) -> TensorVariable:
    """Solve the discrete Lyapunov equation :math:`A X A^H - X = Q`.

    Parameters
    ----------
    A
        Square matrix of shape N x N; must have the same shape as Q
    Q
        Square matrix of shape N x N; must have the same shape as A
    method
        Solver method used, one of ``"direct"`` or ``"bilinear"``. ``"direct"``
        solves the problem directly via matrix inversion.  This has a pure
        Aesara implementation and can thus be cross-compiled to supported
        backends, and should be preferred when ``N`` is not large. The direct
        method scales poorly with the size of ``N``, and the bilinear can be
        used in these cases.

    Returns
    -------
        Square matrix of shape ``N x N``, representing the solution to the
        Lyapunov equation

    """
    if method not in ["direct", "bilinear"]:
        raise ValueError(
            f'Parameter "method" must be one of "direct" or "bilinear", found {method}'
        )

    if method == "direct":
        return _direct_solve_discrete_lyapunov(A, Q)
    if method == "bilinear":
        return _solve_bilinear_direct_lyapunov(A, Q)


def solve_continuous_lyapunov(A: "TensorLike", Q: "TensorLike") -> TensorVariable:
    """Solve the continuous Lyapunov equation :math:`A X + X A^H + Q = 0`.

    Parameters
    ----------
    A
        Square matrix of shape ``N x N``; must have the same shape as `Q`.
    Q
        Square matrix of shape ``N x N``; must have the same shape as `A`.

    Returns
    -------
        Square matrix of shape ``N x N``, representing the solution to the
        Lyapunov equation

    """

    return _solve_continuous_lyapunov(A, Q)


__all__ = [
    "cholesky",
    "solve",
    "eigvalsh",
    "kron",
    "expm",
]

DEPRECATED_NAMES = [
    (
        "solve_lower_triangular",
        "`solve_lower_triangular` is deprecated; use `solve` instead.",
        SolveTriangular(lower=True),
    ),
    (
        "solve_upper_triangular",
        "`solve_upper_triangular` is deprecated; use `solve` instead.",
        SolveTriangular(lower=False),
    ),
    (
        "solve_symmetric",
        "`solve_symmetric` is deprecated; use `solve` instead.",
        Solve(assume_a="sym"),
    ),
]


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + [names[0] for names in DEPRECATED_NAMES])

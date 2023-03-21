from contextlib import ExitStack

import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix

import aesara
import aesara.sparse as sparse
import aesara.tensor as at
from aesara.sparse.type import SparseTensorType
from aesara.tensor.type import DenseTensorType


class TestSparseVariable:
    @pytest.mark.parametrize(
        "method, exp_type, cm, x",
        [
            ("__abs__", DenseTensorType, None, None),
            ("__neg__", SparseTensorType, ExitStack(), None),
            ("__ceil__", DenseTensorType, None, None),
            ("__floor__", DenseTensorType, None, None),
            ("__trunc__", DenseTensorType, None, None),
            ("transpose", DenseTensorType, None, None),
            ("any", DenseTensorType, None, None),
            ("all", DenseTensorType, None, None),
            ("flatten", DenseTensorType, None, None),
            ("ravel", DenseTensorType, None, None),
            ("arccos", DenseTensorType, None, None),
            ("arcsin", DenseTensorType, None, None),
            ("arctan", DenseTensorType, None, None),
            ("arccosh", DenseTensorType, None, None),
            ("arcsinh", DenseTensorType, None, None),
            ("arctanh", DenseTensorType, None, None),
            ("ceil", DenseTensorType, None, None),
            ("cos", DenseTensorType, None, None),
            ("cosh", DenseTensorType, None, None),
            ("deg2rad", DenseTensorType, None, None),
            ("exp", DenseTensorType, None, None),
            ("exp2", DenseTensorType, None, None),
            ("expm1", DenseTensorType, None, None),
            ("floor", DenseTensorType, None, None),
            ("log", DenseTensorType, None, None),
            ("log10", DenseTensorType, None, None),
            ("log1p", DenseTensorType, None, None),
            ("log2", DenseTensorType, None, None),
            ("rad2deg", DenseTensorType, None, None),
            ("sin", DenseTensorType, None, None),
            ("sinh", DenseTensorType, None, None),
            ("sqrt", DenseTensorType, None, None),
            ("tan", DenseTensorType, None, None),
            ("tanh", DenseTensorType, None, None),
            ("copy", DenseTensorType, None, None),
            ("sum", DenseTensorType, ExitStack(), None),
            ("prod", DenseTensorType, None, None),
            ("mean", DenseTensorType, None, None),
            ("var", DenseTensorType, None, None),
            ("std", DenseTensorType, None, None),
            ("min", DenseTensorType, None, None),
            ("max", DenseTensorType, None, None),
            ("argmin", DenseTensorType, None, None),
            ("argmax", DenseTensorType, None, None),
            ("nonzero", DenseTensorType, ExitStack(), None),
            ("nonzero_values", DenseTensorType, None, None),
            ("argsort", DenseTensorType, ExitStack(), None),
            ("conj", SparseTensorType, ExitStack(), at.cmatrix("x")),
            ("round", DenseTensorType, None, None),
            ("trace", DenseTensorType, None, None),
            ("zeros_like", SparseTensorType, ExitStack(), None),
            ("ones_like", DenseTensorType, ExitStack(), None),
            ("cumsum", DenseTensorType, None, None),
            ("cumprod", DenseTensorType, None, None),
            ("ptp", DenseTensorType, None, None),
            ("squeeze", DenseTensorType, None, None),
            ("diagonal", DenseTensorType, None, None),
        ],
    )
    def test_unary(self, method, exp_type, cm, x):
        if x is None:
            x = at.dmatrix("x")

        x = sparse.csr_from_dense(x)

        method_to_call = getattr(x, method)

        if cm is None:
            cm = pytest.warns(UserWarning, match=".*converted to dense.*")

        if exp_type == SparseTensorType:
            exp_res_type = csr_matrix
        else:
            exp_res_type = np.ndarray

        with cm:
            z = method_to_call()

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        assert all(isinstance(out.type, exp_type) for out in z_outs)

        f = aesara.function([x], z, on_unused_input="ignore", allow_input_downcast=True)

        res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        assert all(isinstance(out, exp_res_type) for out in res_outs)

    @pytest.mark.parametrize(
        "method, exp_type",
        [
            ("__lt__", SparseTensorType),
            ("__le__", SparseTensorType),
            ("__gt__", SparseTensorType),
            ("__ge__", SparseTensorType),
            ("__and__", DenseTensorType),
            ("__or__", DenseTensorType),
            ("__xor__", DenseTensorType),
            ("__add__", SparseTensorType),
            ("__sub__", SparseTensorType),
            ("__mul__", SparseTensorType),
            ("__pow__", DenseTensorType),
            ("__mod__", DenseTensorType),
            ("__divmod__", DenseTensorType),
            ("__truediv__", DenseTensorType),
            ("__floordiv__", DenseTensorType),
        ],
    )
    def test_binary(self, method, exp_type):
        x = at.lmatrix("x")
        y = at.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        method_to_call = getattr(x, method)

        if exp_type == SparseTensorType:
            exp_res_type = csr_matrix
            cm = ExitStack()
        else:
            exp_res_type = np.ndarray
            cm = pytest.warns(UserWarning, match=".*converted to dense.*")

        with cm:
            z = method_to_call(y)

        if not isinstance(z, tuple):
            z_outs = (z,)
        else:
            z_outs = z

        assert all(isinstance(out.type, exp_type) for out in z_outs)

        f = aesara.function([x, y], z)
        res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[1, 1, 2], [1, 4, 1]],
        )

        if not isinstance(res, list):
            res_outs = [res]
        else:
            res_outs = res

        assert all(isinstance(out, exp_res_type) for out in res_outs)

    def test_reshape(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.reshape((3, 2))

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_dimshuffle(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.dimshuffle((1, 0))

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

    def test_getitem(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        z = x[:, :2]
        assert isinstance(z.type, SparseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, csr_matrix)

    def test_dot(self):
        x = at.lmatrix("x")
        y = at.lmatrix("y")
        x = sparse.csr_from_dense(x)
        y = sparse.csr_from_dense(y)

        z = x.__dot__(y)
        assert isinstance(z.type, SparseTensorType)

        f = aesara.function([x, y], z)
        exp_res = f(
            [[1, 0, 2], [-1, 0, 0]],
            [[-1], [2], [1]],
        )
        assert isinstance(exp_res, csr_matrix)

    def test_repeat(self):
        x = at.dmatrix("x")
        x = sparse.csr_from_dense(x)

        with pytest.warns(UserWarning, match=".*converted to dense.*"):
            z = x.repeat(2, axis=1)

        assert isinstance(z.type, DenseTensorType)

        f = aesara.function([x], z)
        exp_res = f([[1.1, 0.0, 2.0], [-1.0, 0.0, 0.0]])
        assert isinstance(exp_res, np.ndarray)

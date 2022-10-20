from typing import Iterable, Optional, Union

import numpy as np
import scipy.sparse
from typing_extensions import Literal

import aesara
from aesara import scalar as aes
from aesara.graph.basic import Variable
from aesara.graph.type import HasDataType
from aesara.tensor.type import DenseTensorType, TensorType


SparsityTypes = Literal["csr", "csc", "bsr"]


def _is_sparse(x):
    """

    Returns
    -------
    boolean
        True iff x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray}).

    """
    if not isinstance(x, (scipy.sparse.spmatrix, np.ndarray, tuple, list)):
        raise NotImplementedError(
            "this function should only be called on "
            "sparse.scipy.sparse.spmatrix or "
            "numpy.ndarray, not,",
            x,
        )
    return isinstance(x, scipy.sparse.spmatrix)


class SparseTensorType(TensorType, HasDataType):
    """A `Type` for sparse tensors.

    Notes
    -----
    Currently, sparse tensors can only be matrices (i.e. have two dimensions).

    """

    __props__ = ("dtype", "format", "shape")
    format_cls = {
        "csr": scipy.sparse.csr_matrix,
        "csc": scipy.sparse.csc_matrix,
        "bsr": scipy.sparse.bsr_matrix,
    }
    dtype_specs_map = {
        "float32": (float, "npy_float32", "NPY_FLOAT32"),
        "float64": (float, "npy_float64", "NPY_FLOAT64"),
        "uint8": (int, "npy_uint8", "NPY_UINT8"),
        "int8": (int, "npy_int8", "NPY_INT8"),
        "uint16": (int, "npy_uint16", "NPY_UINT16"),
        "int16": (int, "npy_int16", "NPY_INT16"),
        "uint32": (int, "npy_uint32", "NPY_UINT32"),
        "int32": (int, "npy_int32", "NPY_INT32"),
        "uint64": (int, "npy_uint64", "NPY_UINT64"),
        "int64": (int, "npy_int64", "NPY_INT64"),
        "complex128": (complex, "aesara_complex128", "NPY_COMPLEX128"),
        "complex64": (complex, "aesara_complex64", "NPY_COMPLEX64"),
    }
    ndim = 2

    def __init__(
        self,
        format: SparsityTypes,
        dtype: Union[str, np.dtype],
        shape: Optional[Iterable[Optional[Union[bool, int]]]] = None,
        name: Optional[str] = None,
        broadcastable: Optional[Iterable[bool]] = None,
    ):
        if shape is None and broadcastable is None:
            shape = (None, None)

        if format not in self.format_cls:
            raise ValueError(
                f'unsupported format "{format}" not in list',
            )

        self.format = format

        super().__init__(dtype, shape=shape, name=name, broadcastable=broadcastable)

    def clone(
        self,
        dtype=None,
        shape=None,
        broadcastable=None,
        **kwargs,
    ):
        format: Optional[SparsityTypes] = kwargs.pop("format", self.format)
        if dtype is None:
            dtype = self.dtype
        if shape is None:
            shape = self.shape
        return type(self)(format, dtype, shape=shape, **kwargs)

    def filter(self, value, strict=False, allow_downcast=None):
        if isinstance(value, Variable):
            raise TypeError(
                "Expected an array-like object, but found a Variable: "
                "maybe you are trying to call a function on a (possibly "
                "shared) variable instead of a numeric array?"
            )

        if (
            isinstance(value, self.format_cls[self.format])
            and value.dtype == self.dtype
        ):
            return value

        if strict:
            raise TypeError(
                f"{value} is not sparse, or not the right dtype (is {value.dtype}, "
                f"expected {self.dtype})"
            )

        # The input format could be converted here
        if allow_downcast:
            sp = self.format_cls[self.format](value, dtype=self.dtype)
        else:
            data = self.format_cls[self.format](value)
            up_dtype = aes.upcast(self.dtype, data.dtype)
            if up_dtype != self.dtype:
                raise TypeError(f"Expected {self.dtype} dtype but got {data.dtype}")
            sp = data.astype(up_dtype)

        assert sp.format == self.format

        return sp

    @classmethod
    def may_share_memory(cls, a, b):
        if _is_sparse(a) and _is_sparse(b):
            return (
                cls.may_share_memory(a, b.data)
                or cls.may_share_memory(a, b.indices)
                or cls.may_share_memory(a, b.indptr)
            )
        if _is_sparse(b) and isinstance(a, np.ndarray):
            a, b = b, a
        if _is_sparse(a) and isinstance(b, np.ndarray):
            if (
                np.may_share_memory(a.data, b)
                or np.may_share_memory(a.indices, b)
                or np.may_share_memory(a.indptr, b)
            ):
                # currently we can't share memory with a.shape as it is a tuple
                return True
        return False

    def convert_variable(self, var):
        res = super().convert_variable(var)

        if res is None:
            return res

        if not isinstance(res.type, type(self)):
            if isinstance(res.type, DenseTensorType):
                if self.format == "csr":
                    from aesara.sparse.basic import csr_from_dense

                    return csr_from_dense(res)
                else:
                    from aesara.sparse.basic import csc_from_dense

                    return csc_from_dense(res)

            return None

        if res.format != self.format:
            # TODO: Convert sparse `var`s with different formats to this format?
            return None

        return res

    def __hash__(self):
        return super().__hash__() ^ hash(self.format)

    def __repr__(self):
        return f"Sparse({self.dtype}, {self.shape}, {self.format})"

    def values_eq_approx(self, a, b, eps=1e-6):
        # WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        if not scipy.sparse.issparse(a) or not scipy.sparse.issparse(b):
            return False
        diff = abs(a - b)
        if diff.nnz == 0:
            return True
        # Built-in max from python is not implemented for sparse matrix as a
        # reduction. It returns a sparse matrix which cannot be compared to a
        # scalar. When comparing sparse to scalar, no exceptions is raised and
        # the returning value is not consistent. That is why it is apply to a
        # numpy.ndarray.
        return max(diff.data) < eps

    def values_eq(self, a, b):
        # WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        return (
            scipy.sparse.issparse(a)
            and scipy.sparse.issparse(b)
            and abs(a - b).sum() == 0.0
        )

    def is_valid_value(self, a):
        return scipy.sparse.issparse(a) and (a.format == self.format)

    def get_shape_info(self, obj):
        obj = self.filter(obj)
        assert obj.indices.dtype == "int32"
        assert obj.indptr.dtype == "int32"
        return (obj.shape, obj.data.size, obj.indices.size, obj.indptr.size, obj.nnz)

    def get_size(self, shape_info):
        return (
            shape_info[1] * np.dtype(self.dtype).itemsize
            + (shape_info[2] + shape_info[3]) * np.dtype("int32").itemsize
        )

    def __eq__(self, other):
        res = super().__eq__(other)

        if isinstance(res, bool):
            return res and other.format == self.format

        return res

    def is_super(self, otype):
        if not super().is_super(otype):
            return False

        if self.format == otype.format:
            return True

        return False


aesara.compile.register_view_op_c_code(
    SparseTensorType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)

# This is a deprecated alias used for (temporary) backward-compatibility
SparseType = SparseTensorType

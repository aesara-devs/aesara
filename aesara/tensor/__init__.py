"""Symbolic tensor types and constructor functions."""

import warnings
from functools import singledispatch
from typing import Any, Callable, NoReturn, Optional


def as_tensor_variable(
    x: Any, name: Optional[str] = None, ndim: Optional[int] = None, **kwargs
) -> Callable:
    """Convert `x` into the appropriate ``TensorType``.

    This function is often used by ``make_node`` methods of ``Op`` subclasses
    to turn ndarrays, numbers, ``Scalar`` instances, ``Apply`` instances and
    ``TensorType`` instances into valid input list elements.

    Parameters
    ----------
    x
        The object to be converted into a ``Variable`` type. A
        ``numpy.ndarray`` argument will not be copied, but a list of numbers
        will be copied to make an ``numpy.ndarray``.
    name
        If a new ``Variable`` instance is created, it will be named with this
        string.
    ndim
        Return a ``Variable`` with this many dimensions.
    dtype
        The dtype to use for the resulting ``Variable``.  If `x` is already
        a ``Variable`` type, then the dtype will not be changed.

    Raises
    ------
    TypeError
        If `x` cannot be converted to a ``TensorType`` Variable.

    """
    return _as_tensor_variable(x, name, ndim, **kwargs)


@singledispatch
def _as_tensor_variable(
    x, name: Optional[str], ndim: Optional[int], **kwargs
) -> NoReturn:
    raise NotImplementedError(f"Cannot convert {x} to a tensor variable.")


import aesara.tensor.exceptions
from aesara.gradient import consider_constant, grad, hessian, jacobian
from aesara.tensor import sharedvar  # adds shared-variable constructors
from aesara.tensor import (
    basic_opt,
    blas,
    blas_c,
    blas_scipy,
    nnet,
    opt_uncanonicalize,
    subtensor_opt,
    xlogx,
)


# isort: off
from aesara.tensor import linalg
from aesara.tensor import nlinalg  # For backward compatibility
from aesara.tensor import slinalg  # For backward compatibility

# isort: on
from aesara.tensor.basic import *
from aesara.tensor.blas import batched_dot, batched_tensordot
from aesara.tensor.extra_ops import (
    bartlett,
    bincount,
    broadcast_arrays,
    broadcast_shape,
    broadcast_shape_iter,
    broadcast_to,
    cumprod,
    cumsum,
    diff,
    fill_diagonal,
    fill_diagonal_offset,
    ravel_multi_index,
    repeat,
    squeeze,
    unique,
    unravel_index,
)
from aesara.tensor.io import *
from aesara.tensor.math import *
from aesara.tensor.shape import (
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    specify_shape,
)

# We import as `_shared` instead of `shared` to avoid confusion between
# `aesara.shared` and `tensor._shared`.
from aesara.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk
from aesara.tensor.subtensor import *
from aesara.tensor.type import *
from aesara.tensor.type_other import *
from aesara.tensor.var import TensorConstant, TensorVariable


__all__ = ["random"]  # noqa: F405

"""Symbolic tensor types and constructor functions."""

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Optional, Sequence, Union

from aesara.graph.basic import Constant, Variable
from aesara.graph.op import Op


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


TensorLike = Union[Variable, Sequence[Variable], "ArrayLike"]


def as_tensor_variable(
    x: TensorLike, name: Optional[str] = None, ndim: Optional[int] = None, **kwargs
) -> "TensorVariable":
    """Convert `x` into an equivalent `TensorVariable`.

    This function can be used to turn ndarrays, numbers, `ScalarType` instances,
    `Apply` instances and `TensorVariable` instances into valid input list
    elements.

    See `aesara.as_symbolic` for a more general conversion function.

    Parameters
    ----------
    x
        The object to be converted into a `Variable` type. A
        `numpy.ndarray` argument will not be copied, but a list of numbers
        will be copied to make an `numpy.ndarray`.
    name
        If a new `Variable` instance is created, it will be named with this
        string.
    ndim
        Return a `Variable` with this many dimensions.
    dtype
        The dtype to use for the resulting `Variable`.  If `x` is already
        a `Variable` type, then the dtype will not be changed.

    Raises
    ------
    TypeError
        If `x` cannot be converted to a `TensorVariable`.

    """
    return _as_tensor_variable(x, name, ndim, **kwargs)


@singledispatch
def _as_tensor_variable(
    x: TensorLike, name: Optional[str], ndim: Optional[int], **kwargs
) -> "TensorVariable":
    raise NotImplementedError(f"Cannot convert {x!r} to a tensor variable.")


def get_vector_length(v: TensorLike) -> int:
    """Return the run-time length of a symbolic vector, when possible.

    Parameters
    ----------
    v
        A rank-1 `TensorType` variable.

    Raises
    ------
    TypeError
        `v` hasn't the proper type.
    ValueError
        No special case applies, the length is not known.
        In general this is not possible, but for a number of special cases
        the length can be determined at compile / graph-construction time.
        This function implements these special cases.

    """
    v = as_tensor_variable(v)

    if v.type.ndim != 1:
        raise TypeError(f"Argument must be a vector; got {v.type}")

    static_shape: Optional[int] = v.type.shape[0]
    if static_shape is not None:
        return static_shape

    return _get_vector_length(getattr(v.owner, "op", v), v)


@singledispatch
def _get_vector_length(op: Union[Op, Variable], var: Variable) -> int:
    """`Op`-based dispatch for `get_vector_length`."""
    raise ValueError(f"Length of {var} cannot be determined")


@_get_vector_length.register(Constant)
def _get_vector_length_Constant(op: Union[Op, Variable], var: Constant) -> int:
    return len(var.data)


import aesara.tensor.exceptions  # noqa
from aesara.gradient import consider_constant, grad, hessian, jacobian  # noqa

# adds shared-variable constructors
from aesara.tensor import sharedvar  # noqa
from aesara.tensor import (  # noqa
    blas,
    blas_c,
    blas_scipy,
    xlogx,
)
import aesara.tensor.rewriting


# isort: off
from aesara.tensor import linalg  # noqa
from aesara.tensor import special

# For backward compatibility
from aesara.tensor import nlinalg  # noqa
from aesara.tensor import slinalg  # noqa

# isort: on
from aesara.tensor.basic import *  # noqa
from aesara.tensor.blas import batched_dot, batched_tensordot  # noqa
from aesara.tensor.extra_ops import *


from aesara.tensor.shape import (  # noqa
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    specify_shape,
)


from aesara.tensor.io import *  # noqa
from aesara.tensor.math import *  # noqa

# We import as `_shared` instead of `shared` to avoid confusion between
# `aesara.shared` and `tensor._shared`.
from aesara.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk  # noqa
from aesara.tensor.subtensor import *  # noqa
from aesara.tensor.type import *  # noqa
from aesara.tensor.type_other import *  # noqa
from aesara.tensor.var import TensorConstant, TensorVariable  # noqa


__all__ = ["random"]  # noqa: F405

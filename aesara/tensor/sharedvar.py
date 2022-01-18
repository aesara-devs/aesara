import traceback
import warnings

import numpy as np

from aesara.compile import SharedVariable, shared_constructor
from aesara.misc.safe_asarray import _asarray
from aesara.tensor import _get_vector_length
from aesara.tensor.type import TensorType
from aesara.tensor.var import _tensor_py_operators


def load_shared_variable(val):
    """
    This function is only here to keep some pickles loading
    after a failed fix done in August 2011.
    It can be removed after sufficient time has passed.

    """
    return tensor_constructor(val)


# _tensor_py_operators is first to have its version of __{gt,ge,lt,le}__
class TensorSharedVariable(_tensor_py_operators, SharedVariable):
    pass


@_get_vector_length.register(TensorSharedVariable)
def _get_vector_length_TensorSharedVariable(var_inst, var):
    return len(var.get_value(borrow=True))


@shared_constructor
def tensor_constructor(
    value,
    name=None,
    strict=False,
    allow_downcast=None,
    borrow=False,
    shape=None,
    target="cpu",
    broadcastable=None,
):
    """
    SharedVariable Constructor for TensorType.

    Notes
    -----
    The default is to assume that the `shape` value might be resized in any
    dimension, so the default shape is ``(None,) * len(value.shape)``.  The
    optional `shape` argument will override this default.

    """
    if broadcastable is not None:
        warnings.warn(
            "The `broadcastable` keyword is deprecated; use `shape`.",
            DeprecationWarning,
        )
        shape = broadcastable

    if target != "cpu":
        raise TypeError("not for cpu")

    if not isinstance(value, np.ndarray):
        raise TypeError()

    # if no shape is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if shape is None:
        shape = (False,) * len(value.shape)
    type = TensorType(value.dtype, shape=shape)
    return TensorSharedVariable(
        type=type,
        value=np.array(value, copy=(not borrow)),
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
    )


# TensorSharedVariable brings in the tensor operators, is not ideal, but works
# as long as we don't do purely scalar-scalar operations
# _tensor_py_operators is first to have its version of __{gt,ge,lt,le}__
#
# N.B. THERE IS ANOTHER CLASS CALLED ScalarSharedVariable in the
# aesara.scalar.sharedvar file.  It is not registered as a shared_constructor,
# this one is.
class ScalarSharedVariable(_tensor_py_operators, SharedVariable):
    pass


@shared_constructor
def scalar_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False, target="cpu"
):
    """
    SharedVariable constructor for scalar values. Default: int64 or float64.

    Notes
    -----
    We implement this using 0-d tensors for now.

    We ignore the borrow parameter as we convert ``value`` to an
    ndarray (this is a new object). This respects the semantic of
    borrow, as it is a hint to Aesara that we can reuse it.

    """
    if target != "cpu":
        raise TypeError("not for cpu")

    if not isinstance(value, (np.number, float, int, complex)):
        raise TypeError()
    try:
        dtype = value.dtype
    except Exception:
        dtype = np.asarray(value).dtype

    dtype = str(dtype)
    value = _asarray(value, dtype=dtype)
    tensor_type = TensorType(dtype=str(value.dtype), shape=[])

    try:
        # Do not pass the dtype to asarray because we want this to fail if
        # strict is True and the types do not match.
        rval = ScalarSharedVariable(
            type=tensor_type,
            value=np.array(value, copy=True),
            name=name,
            strict=strict,
            allow_downcast=allow_downcast,
        )
        return rval
    except Exception:
        traceback.print_exc()
        raise

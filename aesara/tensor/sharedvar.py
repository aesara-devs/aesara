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


class TensorSharedVariable(_tensor_py_operators, SharedVariable):
    def zero(self, borrow: bool = False):
        r"""Set the values of a shared variable to 0.

        Parameters
        ----------
        borrow
            ``True`` to modify the value of a shared variable directly by using
            its previous value. Potentially this can cause problems regarding
            to the aliased memory.

        Changes done with this function will be visible to all functions using
        this `SharedVariable`.

        """
        if borrow:
            self.container.value[...] = 0
        else:
            self.container.value = 0 * self.container.value


@_get_vector_length.register(TensorSharedVariable)
def _get_vector_length_TensorSharedVariable(var_inst, var):
    return len(var.get_value(borrow=True))


@shared_constructor.register(np.ndarray)
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
    r"""`SharedVariable` constructor for `TensorType`\s.

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

    # If no shape is given, then the default is to assume that the value might
    # be resized in any dimension in the future.
    if shape is None:
        shape = (None,) * value.ndim

    type = TensorType(value.dtype, shape=shape)

    return TensorSharedVariable(
        type=type,
        value=np.array(value, copy=(not borrow)),
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )


class ScalarSharedVariable(TensorSharedVariable):
    pass


@shared_constructor.register(np.number)
@shared_constructor.register(float)
@shared_constructor.register(int)
@shared_constructor.register(complex)
def scalar_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False, target="cpu"
):
    """`SharedVariable` constructor for scalar values.

    Default: int64 or float64.

    Notes
    -----
    We implement this using 0-d tensors for now.

    We ignore the borrow parameter as we convert ``value`` to an
    ndarray (this is a new object). This respects the semantic of
    borrow, as it is a hint to Aesara that we can reuse it.

    """
    if target != "cpu":
        raise TypeError("not for cpu")

    try:
        dtype = value.dtype
    except AttributeError:
        dtype = np.asarray(value).dtype

    dtype = str(dtype)
    value = _asarray(value, dtype=dtype)
    tensor_type = TensorType(dtype=str(value.dtype), shape=())

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

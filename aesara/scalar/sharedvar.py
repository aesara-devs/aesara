"""
A shared variable container for true scalars - for internal use.

Why does this file exist?
-------------------------

Scalars are used to index subtensors.  Subtensor indexing is the heart of what
looks like the new scan mechanism.  This little file made it possible to catch
up to the Python interpreter in benchmarking tests.

We don't want to encourage people to use scalars (rather than 0-d tensors), but
the reason is just to keep the docs simple, not because scalars are bad.  If we
just don't register this shared variable constructor to handle any values by
default when calling aesara.shared(value) then users must really go out of their
way (as scan does) to create a shared variable of this kind.

"""

import numpy as np

from aesara.compile import SharedVariable
from aesara.scalar.basic import ScalarType, _scalar_py_operators


class ScalarSharedVariable(_scalar_py_operators, SharedVariable):
    pass


# this is not installed in the default shared variable registry so that
# scalars are typically 0-d tensors.
# still, in case you need a shared variable scalar, you can get one
# by calling this function directly.


def shared(value, name=None, strict=False, allow_downcast=None):
    """
    SharedVariable constructor for scalar values. Default: int64 or float64.

    Notes
    -----
    We implement this using 0-d tensors for now.

    """
    if not isinstance(value, (np.number, float, int, complex)):
        raise TypeError()
    try:
        dtype = value.dtype
    except AttributeError:
        dtype = np.asarray(value).dtype

    dtype = str(dtype)
    value = getattr(np, dtype)(value)
    scalar_type = ScalarType(dtype=dtype)
    rval = ScalarSharedVariable(
        type=scalar_type,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
    )
    return rval

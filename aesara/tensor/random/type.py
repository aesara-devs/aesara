import sys

import numpy as np

import aesara
from aesara.graph.type import Type


class RandomStateType(Type):
    """A Type wrapper for `numpy.random.RandomState`.

    The reason this exists (and `Generic` doesn't suffice) is that
    `RandomState` objects that would appear to be equal do not compare equal
    with the `==` operator.  This `Type` exists to provide an equals function
    that is used by `DebugMode`.

    Also works with a `dict` derived from RandomState.get_state() unless
    the `strict` argument is explicitly set to `True`.

    """

    def __repr__(self):
        return "RandomStateType"

    @classmethod
    def filter(cls, data, strict=False, allow_downcast=None):
        if cls.is_valid_value(data, strict):
            return data
        else:
            raise TypeError()

    @staticmethod
    def is_valid_value(a, strict):
        if isinstance(a, np.random.RandomState):
            return True

        if not strict and isinstance(a, dict):
            gen_keys = ["bit_generator", "gauss", "has_gauss", "state"]
            state_keys = ["key", "pos"]

            for key in gen_keys:
                if key not in a:
                    return False

            for key in state_keys:
                if key not in a["state"]:
                    return False

            state_key = a["state"]["key"]
            if state_key.shape == (624,) and state_key.dtype == np.uint32:
                return True

        return False

    @staticmethod
    def values_eq(a, b):
        sa = a if isinstance(a, dict) else a.get_state(legacy=False)
        sb = b if isinstance(b, dict) else b.get_state(legacy=False)

        def _eq(sa, sb):
            for key in sa:
                if isinstance(sa[key], dict):
                    if not _eq(sa[key], sb[key]):
                        return False
                elif isinstance(sa[key], np.ndarray):
                    if not np.array_equal(sa[key], sb[key]):
                        return False
                else:
                    if sa[key] != sb[key]:
                        return False

            return True

        return _eq(sa, sb)

    @staticmethod
    def get_shape_info(obj):
        return obj.get_value(borrow=True)

    @staticmethod
    def get_size(shape_info):
        return sys.getsizeof(shape_info.get_state(legacy=False))

    @staticmethod
    def may_share_memory(a, b):
        return a._bit_generator is b._bit_generator


# Register `RandomStateType`'s C code for `ViewOp`.
aesara.compile.register_view_op_c_code(
    RandomStateType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)

random_state_type = RandomStateType()

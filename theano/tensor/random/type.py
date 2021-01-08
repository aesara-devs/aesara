import sys

import numpy as np

import theano
from theano.graph.type import Type


class RandomStateType(Type):
    """A Type wrapper for `numpy.random.RandomState`.

    The reason this exists (and `Generic` doesn't suffice) is that
    `RandomState` objects that would appear to be equal do not compare equal
    with the `==` operator.  This `Type` exists to provide an equals function
    that is used by `DebugMode`.

    """

    def __repr__(self):
        return "RandomStateType"

    @classmethod
    def filter(cls, data, strict=False, allow_downcast=None):
        if cls.is_valid_value(data):
            return data
        else:
            raise TypeError()

    @staticmethod
    def is_valid_value(a):
        return isinstance(a, np.random.RandomState)

    @staticmethod
    def values_eq(a, b):
        sa = a.get_state(legacy=False)
        sb = b.get_state(legacy=False)

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
theano.compile.register_view_op_c_code(
    RandomStateType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)

random_state_type = RandomStateType()

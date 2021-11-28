import numpy as np

import aesara
from aesara.graph.type import Type


gen_states_keys = {
    "MT19937": (["state"], ["key", "pos"]),
    "PCG64": (["state", "has_uint32", "uinteger"], ["state", "inc"]),
    "Philox": (
        ["state", "buffer", "buffer_pos", "has_uint32", "uinteger"],
        ["counter", "key"],
    ),
    "SFC64": (["state", "has_uint32", "uinteger"], ["state"]),
}

# We map bit generators to an integer index so that we can avoid using strings
numpy_bit_gens = {0: "MT19937", 1: "PCG64", 2: "Philox", 3: "SFC64"}


class RandomType(Type):
    r"""A Type wrapper for `numpy.random.Generator` and `numpy.random.RandomState`."""

    @classmethod
    def filter(cls, data, strict=False, allow_downcast=None):
        if cls.is_valid_value(data, strict):
            return data
        else:
            raise TypeError()

    @staticmethod
    def may_share_memory(a, b):
        return a._bit_generator is b._bit_generator


class RandomStateType(RandomType):
    r"""A Type wrapper for `numpy.random.RandomState`.

    The reason this exists (and `Generic` doesn't suffice) is that
    `RandomState` objects that would appear to be equal do not compare equal
    with the ``==`` operator.

    This `Type` also works with a ``dict`` derived from
    `RandomState.get_state(legacy=False)`, unless the ``strict`` argument to `Type.filter`
    is explicitly set to ``True``.

    """

    def __repr__(self):
        return "RandomStateType"

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

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


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


class RandomGeneratorType(RandomType):
    r"""A Type wrapper for `numpy.random.Generator`.

    The reason this exists (and `Generic` doesn't suffice) is that
    `Generator` objects that would appear to be equal do not compare equal
    with the ``==`` operator.

    This `Type` also works with a ``dict`` derived from
    `Generator.__get_state__`, unless the ``strict`` argument to `Type.filter`
    is explicitly set to ``True``.

    """

    def __repr__(self):
        return "RandomGeneratorType"

    @staticmethod
    def is_valid_value(a, strict):
        if isinstance(a, np.random.Generator):
            return True

        if not strict and isinstance(a, dict):
            if "bit_generator" not in a:
                return False
            else:
                bit_gen_key = a["bit_generator"]

                if hasattr(bit_gen_key, "_value"):
                    bit_gen_key = int(bit_gen_key._value)
                    bit_gen_key = numpy_bit_gens[bit_gen_key]

                gen_keys, state_keys = gen_states_keys[bit_gen_key]

                for key in gen_keys:
                    if key not in a:
                        return False

                for key in state_keys:
                    if key not in a["state"]:
                        return False

                return True

        return False

    @staticmethod
    def values_eq(a, b):
        sa = a if isinstance(a, dict) else a.__getstate__()
        sb = b if isinstance(b, dict) else b.__getstate__()

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

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


# Register `RandomGeneratorType`'s C code for `ViewOp`.
aesara.compile.register_view_op_c_code(
    RandomGeneratorType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)

random_generator_type = RandomGeneratorType()

from typing import TypeVar

import numpy as np

import aesara
from aesara.graph.type import Type


T = TypeVar("T", np.random.RandomState, np.random.Generator)


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


class RandomType(Type[T]):
    r"""A Type wrapper for `numpy.random.Generator` and `numpy.random.RandomState`."""

    @staticmethod
    def may_share_memory(a: T, b: T):
        return a._bit_generator is b._bit_generator  # type: ignore[attr-defined]


class RandomStateType(RandomType[np.random.RandomState]):
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

    def filter(self, data, strict: bool = False, allow_downcast=None):
        """
        XXX: This doesn't convert `data` to the same type of underlying RNG type
        as `self`.  It really only checks that `data` is of the appropriate type
        to be a valid `RandomStateType`.

        In other words, it serves as a `Type.is_valid_value` implementation,
        but, because the default `Type.is_valid_value` depends on
        `Type.filter`, we need to have it here to avoid surprising circular
        dependencies in sub-classes.
        """
        if isinstance(data, np.random.RandomState):
            return data

        if not strict and isinstance(data, dict):
            gen_keys = ["bit_generator", "gauss", "has_gauss", "state"]
            state_keys = ["key", "pos"]

            for key in gen_keys:
                if key not in data:
                    raise TypeError()

            for key in state_keys:
                if key not in data["state"]:
                    raise TypeError()

            state_key = data["state"]["key"]
            if state_key.shape == (624,) and state_key.dtype == np.uint32:
                # TODO: Add an option to convert to a `RandomState` instance?
                return data

        raise TypeError()

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


class RandomGeneratorType(RandomType[np.random.Generator]):
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

    def filter(self, data, strict=False, allow_downcast=None):
        """
        XXX: This doesn't convert `data` to the same type of underlying RNG type
        as `self`.  It really only checks that `data` is of the appropriate type
        to be a valid `RandomGeneratorType`.

        In other words, it serves as a `Type.is_valid_value` implementation,
        but, because the default `Type.is_valid_value` depends on
        `Type.filter`, we need to have it here to avoid surprising circular
        dependencies in sub-classes.
        """
        if isinstance(data, np.random.Generator):
            return data

        if not strict and isinstance(data, dict):
            if "bit_generator" not in data:
                raise TypeError()
            else:
                bit_gen_key = data["bit_generator"]

                if hasattr(bit_gen_key, "_value"):
                    bit_gen_key = int(bit_gen_key._value)
                    bit_gen_key = numpy_bit_gens[bit_gen_key]

                gen_keys, state_keys = gen_states_keys[bit_gen_key]

                for key in gen_keys:
                    if key not in data:
                        raise TypeError()

                for key in state_keys:
                    if key not in data["state"]:
                        raise TypeError()

                return data

        raise TypeError()

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

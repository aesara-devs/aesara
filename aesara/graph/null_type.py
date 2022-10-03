from aesara.graph.type import NewTypeMeta, Props, Type


class NullTypeMeta(NewTypeMeta):
    """
    A type that allows no values.

    Used to represent expressions
    that are undefined, either because they do not exist mathematically
    or because the code to generate the expression has not been
    implemented yet.

    Parameters
    ----------
    why_null : str
        A string explaining why this variable can't take on any values.

    """

    why_null: Props[str] = None

    def filter(self, data, strict=False, allow_downcast=None):
        raise ValueError("No values may be assigned to a NullType")

    def filter_variable(self, other, allow_convert=True):
        raise ValueError("No values may be assigned to a NullType")

    def may_share_memory(a, b):
        return False

    def values_eq(self, a, b, force_same_dtype=True):
        raise ValueError("NullType has no values to compare")

    def __str__(self):
        return "NullType"


class NullType(Type, metaclass=NullTypeMeta):
    pass


null_type = NullType.subtype()

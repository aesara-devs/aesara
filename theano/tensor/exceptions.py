class ShapeError(Exception):
    """Raised when the shape cannot be computed."""


class NotScalarConstantError(Exception):
    """
    Raised by get_scalar_constant_value if called on something that is
    not a scalar constant.
    """


class EmptyConstantError(NotScalarConstantError):
    """
    Raised by get_scalar_const_value if called on something that is a
    zero dimensional constant.
    """


class AdvancedIndexingError(TypeError):
    """
    Raised when Subtensor is asked to perform advanced indexing.

    """

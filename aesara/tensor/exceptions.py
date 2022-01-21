class ShapeError(Exception):
    """Raised when the shape cannot be computed."""


class AdvancedIndexingError(TypeError):
    """
    Raised when Subtensor is asked to perform advanced indexing.

    """

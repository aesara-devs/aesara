import warnings


warnings.warn(
    "The module `aesara.assert_op` is deprecated "
    "and its `Op`s have been moved to `aesara.raise_op`",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.raise_op import Assert, assert_op  # noqa: F401 E402

import warnings


warnings.warn(
    "The module `aesara.tensor.math_opt` is deprecated; use `aesara.tensor.rewriting.math` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.rewriting.math import *  # noqa: F401 E402 F403

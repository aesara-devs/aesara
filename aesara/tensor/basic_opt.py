import warnings


warnings.warn(
    "The module `aesara.tensor.basic_opt` is deprecated; use `aesara.tensor.rewriting.basic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.rewriting.basic import *  # noqa: F401 E402 F403

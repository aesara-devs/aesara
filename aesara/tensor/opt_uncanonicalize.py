import warnings


warnings.warn(
    "The module `aesara.tensor.opt_uncanonicalize` is deprecated; use `aesara.tensor.rewriting.uncanonicalize` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.rewriting.uncanonicalize import *  # noqa: F401 E402 F403

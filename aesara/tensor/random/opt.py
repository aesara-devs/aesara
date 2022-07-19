import warnings


warnings.warn(
    "The module `aesara.tensor.random.opt` is deprecated; use `aesara.tensor.random.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.random.rewriting import *  # noqa: F401 E402 F403

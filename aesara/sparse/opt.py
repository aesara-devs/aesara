import warnings


warnings.warn(
    "The module `aesara.sparse.opt` is deprecated; use `aesara.sparse.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.sparse.rewriting import *  # noqa: F401 E402 F403

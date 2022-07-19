import warnings


warnings.warn(
    "The module `aesara.graph.kanren` is deprecated; use `aesara.graph.rewriting.kanren` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.graph.rewriting.kanren import *  # noqa: F401 E402 F403

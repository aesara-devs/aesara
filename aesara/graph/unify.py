import warnings


warnings.warn(
    "The module `aesara.graph.unify` is deprecated; use `aesara.graph.rewriting.unify` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.graph.rewriting.unify import *  # noqa: F401 E402 F403

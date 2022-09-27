import warnings


warnings.warn(
    "The module `aesara.tensor.nnet.opt` is deprecated; use `aesara.tensor.nnet.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.nnet.rewriting import *  # noqa: F401 E402 F403

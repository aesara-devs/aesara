import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import warnings


warnings.warn(
    "The module `aesara.scan.opt` is deprecated; use `aesara.scan.rewriting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.scan.rewriting import *  # noqa: F401 E402 F403

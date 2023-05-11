import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import warnings


warnings.warn(
    "The module `aesara.tensor.subtracttensor_opt` is deprecated; use `aesara.tensor.rewriting.subtensor` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.rewriting.subtensor import *  # noqa: F401 E402 F403

import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import warnings


warnings.warn(
    "The module `aesara.scalar.basic_scipy` is deprecated "
    "and has been renamed to `aesara.scalar.math`",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.scalar.math import *

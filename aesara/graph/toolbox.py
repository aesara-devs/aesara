import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import warnings


warnings.warn(
    "The module `aesara.graph.toolbox` is deprecated "
    "and has been renamed to `aesara.graph.features`",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.graph.toolbox import *

import warnings


warnings.warn(
    "The module `aesara.tensor.basic_opt` is deprecated; use `aesara.tensor.rewriting.basic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.tensor.rewriting.basic import *  # noqa: F401 E402 F403
from aesara.tensor.rewriting.elemwise import *  # noqa: F401 E402 F403
from aesara.tensor.rewriting.extra_ops import *  # noqa: F401 E402 F403
from aesara.tensor.rewriting.shape import *  # noqa: F401 E402 F403

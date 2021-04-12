import warnings


warnings.warn(
    "The module `aesara.link.jax.jax_dispatch` is deprecated "
    "and has been renamed to `aesara.link.jax.dispatch`",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.link.jax.dispatch import *

import warnings


warnings.warn(
    "The module `aesara.link.jax.jax_linker` is deprecated "
    "and has been renamed to `aesara.link.jax.linker`",
    DeprecationWarning,
    stacklevel=2,
)

from aesara.link.jax.linker import *

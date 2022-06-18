import warnings


warnings.warn(
    "The module `aesara.tensor.basic_opt` is deprecated; use `aesara.tensor.rewriting.basic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

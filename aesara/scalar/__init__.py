from typing import List, Tuple

from aesara.scalar.basic import *
from aesara.scalar.math import *


# isort: off
from aesara.scalar.basic import DEPRECATED_NAMES as BASIC_DEPRECATIONS

# isort: on

DEPRECATED_NAMES: List[Tuple[str, str, object]] = BASIC_DEPRECATIONS


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")

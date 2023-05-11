import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
# TODO: This is for backward-compatibility; remove when reasonable.
from aesara.tensor.random.rewriting.basic import *


# isort: off

# Register JAX specializations
import aesara.tensor.random.rewriting.jax

# isort: on

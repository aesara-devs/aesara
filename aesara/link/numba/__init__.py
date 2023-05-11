import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
from aesara.link.numba.linker import NumbaLinker

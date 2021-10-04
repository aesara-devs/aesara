# isort: off
from aesara.link.numba.dispatch.basic import numba_funcify, numba_typify

# Load dispatch specializations
import aesara.link.numba.dispatch.scalar
import aesara.link.numba.dispatch.tensor_basic
import aesara.link.numba.dispatch.extra_ops
import aesara.link.numba.dispatch.nlinalg
import aesara.link.numba.dispatch.random
import aesara.link.numba.dispatch.elemwise
import aesara.link.numba.dispatch.scan

# isort: on

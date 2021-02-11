"""
Aesara is an optimizing compiler in Python, built to evaluate
complicated expressions (especially matrix-valued ones) as quickly as
possible.  Aesara compiles expression graphs (see :doc:`graph` ) that
are built by Python code. The expressions in these graphs are called
`Apply` nodes and the variables in these graphs are called `Variable`
nodes.

You compile a graph by calling `function`, which takes a graph, and
returns a callable object.  One of aesara's most important features is
that `function` can transform your graph before compiling it.  It can
replace simple expressions with faster or more numerically stable
implementations.

To learn more, check out:

- Op List (:doc:`oplist`)

"""


__docformat__ = "restructuredtext en"

# Set a default logger. It is important to do this before importing some other
# aesara code, since this code may want to log some messages.
import logging
import os
import sys


aesara_logger = logging.getLogger("aesara")
logging_default_handler = logging.StreamHandler()
logging_default_formatter = logging.Formatter(
    fmt="%(levelname)s (%(name)s): %(message)s"
)
logging_default_handler.setFormatter(logging_default_formatter)
aesara_logger.setLevel(logging.WARNING)

if not aesara_logger.hasHandlers():
    aesara_logger.addHandler(logging_default_handler)


# Disable default log handler added to aesara_logger when the module
# is imported.
def disable_log_handler(logger=aesara_logger, handler=logging_default_handler):
    if logger.hasHandlers():
        logger.removeHandler(handler)


# Version information.
from aesara.version import version as __version__


# Raise a meaningful warning/error if the aesara directory is in the Python
# path.
rpath = os.path.realpath(__path__[0])
for p in sys.path:
    if os.path.realpath(p) != rpath:
        continue
    raise RuntimeError("You have the aesara directory in your Python path.")

from aesara.configdefaults import config
from aesara.utils import deprecated


change_flags = deprecated("Use aesara.config.change_flags instead!")(
    config.change_flags
)


# This is the api version for ops that generate C code.  External ops
# might need manual changes if this number goes up.  An undefined
# __api_version__ can be understood to mean api version 0.
#
# This number is not tied to the release version and should change
# very rarely.
__api_version__ = 1

from aesara import scalar, tensor
from aesara.compile import (
    In,
    Mode,
    Out,
    Param,
    ProfileStats,
    predefined_linkers,
    predefined_modes,
    predefined_optimizers,
    shared,
)
from aesara.compile.function import function, function_dump
from aesara.compile.function.types import FunctionMaker
from aesara.gradient import Lop, Rop, grad, subgraph_grad
from aesara.printing import debugprint as dprint
from aesara.printing import pp, pprint
from aesara.updates import OrderedUpdates


if (
    config.device.startswith("cuda")
    or config.device.startswith("opencl")
    or config.init_gpu_device.startswith("cuda")
    or config.init_gpu_device.startswith("opencl")
    or config.contexts != ""
):
    import aesara.gpuarray


def get_scalar_constant_value(v):
    """Return the constant scalar (i.e. 0-D) value underlying variable `v`.

    If `v` is the output of dim-shuffles, fills, allocs, rebroadcasts, cast
    this function digs through them.

    If ``aesara.sparse`` is also there, we will look over CSM `Op`.

    If `v` is not some view of constant data, then raise a
    `NotScalarConstantError`.
    """
    # Is it necessary to test for presence of aesara.sparse at runtime?
    sparse = globals().get("sparse")
    if sparse and isinstance(v.type, sparse.SparseType):
        if v.owner is not None and isinstance(v.owner.op, sparse.CSM):
            data = v.owner.inputs[0]
            return tensor.get_scalar_constant_value(data)
    return tensor.get_scalar_constant_value(v)


import aesara.tensor.random.var
from aesara.graph.basic import clone_replace
from aesara.scan import checkpoints
from aesara.scan.basic import scan
from aesara.scan.views import foldl, foldr, map, reduce


# Some config variables are registered by submodules. Only after all those imports
# were executed, we can warn about remaining flags provided by the user through AESARA_FLAGS.
config.warn_unused_flags()

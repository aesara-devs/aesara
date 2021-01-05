"""
Theano is an optimizing compiler in Python, built to evaluate
complicated expressions (especially matrix-valued ones) as quickly as
possible.  Theano compiles expression graphs (see :doc:`graph` ) that
are built by Python code. The expressions in these graphs are called
`Apply` nodes and the variables in these graphs are called `Variable`
nodes.

You compile a graph by calling `function`, which takes a graph, and
returns a callable object.  One of theano's most important features is
that `function` can transform your graph before compiling it.  It can
replace simple expressions with faster or more numerically stable
implementations.

To learn more, check out:

- Op List (:doc:`oplist`)

The markup language used in the docstrings is ReStructured Text,
which may be rendered with Sphinx. A rendered version is
maintained at http://www.deeplearning.net/software/theano/library/

"""


__docformat__ = "restructuredtext en"

# Set a default logger. It is important to do this before importing some other
# theano code, since this code may want to log some messages.
import logging
import os
import sys


theano_logger = logging.getLogger("theano")
logging_default_handler = logging.StreamHandler()
logging_default_formatter = logging.Formatter(
    fmt="%(levelname)s (%(name)s): %(message)s"
)
logging_default_handler.setFormatter(logging_default_formatter)
theano_logger.setLevel(logging.WARNING)

if not theano_logger.hasHandlers():
    theano_logger.addHandler(logging_default_handler)


# Disable default log handler added to theano_logger when the module
# is imported.
def disable_log_handler(logger=theano_logger, handler=logging_default_handler):
    if logger.hasHandlers():
        logger.removeHandler(handler)


# Version information.
from theano.version import version as __version__


# Raise a meaningful warning/error if the theano directory is in the Python
# path.
rpath = os.path.realpath(__path__[0])
for p in sys.path:
    if os.path.realpath(p) != rpath:
        continue
    raise RuntimeError("You have the theano directory in your Python path.")

from theano.configdefaults import config
from theano.utils import deprecated


change_flags = deprecated("Use theano.config.change_flags instead!")(
    config.change_flags
)


# This is the api version for ops that generate C code.  External ops
# might need manual changes if this number goes up.  An undefined
# __api_version__ can be understood to mean api version 0.
#
# This number is not tied to the release version and should change
# very rarely.
__api_version__ = 1

from theano import scalar, tensor
from theano.compile import (
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
from theano.compile.function import function, function_dump
from theano.compile.function.types import FunctionMaker
from theano.gradient import Lop, Rop, grad, subgraph_grad
from theano.printing import pp, pprint
from theano.updates import OrderedUpdates


if (
    config.device.startswith("cuda")
    or config.device.startswith("opencl")
    or config.init_gpu_device.startswith("cuda")
    or config.init_gpu_device.startswith("opencl")
    or config.contexts != ""
):
    import theano.gpuarray


def get_scalar_constant_value(v):
    """Return the constant scalar (i.e. 0-D) value underlying variable `v`.

    If v is the output of dimshuffles, fills, allocs, rebroadcasts, cast
    this function digs through them.

    If theano.sparse is also there, we will look over CSM op.

    If `v` is not some view of constant data, then raise a
    tensor.basic.NotScalarConstantError.
    """
    # Is it necessary to test for presence of theano.sparse at runtime?
    sparse = globals().get("sparse")
    if sparse and isinstance(v.type, sparse.SparseType):
        if v.owner is not None and isinstance(v.owner.op, sparse.CSM):
            data = v.owner.inputs[0]
            return tensor.get_scalar_constant_value(data)
    return tensor.get_scalar_constant_value(v)


def sparse_grad(var):
    """This function return a new variable whose gradient will be
    stored in a sparse format instead of dense.

    Currently only variable created by AdvancedSubtensor1 is supported.
    i.e. a_tensor_var[an_int_vector].

    .. versionadded:: 0.6rc4
    """
    assert isinstance(var.owner.op, tensor.AdvancedSubtensor1)
    ret = var.owner.op.__class__(sparse_grad=True)(*var.owner.inputs)
    return ret


import theano.tensor.random.var
from theano.scan import checkpoints, clone, foldl, foldr, map, reduce, scan


# Some config variables are registered by submodules. Only after all those imports
# were executed, we can warn about remaining flags provided by the user through THEANO_FLAGS.
config.warn_unused_flags()

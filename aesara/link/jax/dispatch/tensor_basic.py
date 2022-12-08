import jax.numpy as jnp

from aesara.graph.basic import Constant
from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
)


ARANGE_CONCRETE_VALUE_ERROR = """JAX requires the arguments of `jax.numpy.arange`
to be constants. The graph that you defined thus cannot be JIT-compiled
by JAX. An example of a graph that can be compiled to JAX:

>>> import aesara.tensor basic
>>> at.arange(1, 10, 2)
"""


@jax_funcify.register(AllocDiag)
def jax_funcify_AllocDiag(op, **kwargs):
    offset = op.offset

    def allocdiag(v, offset=offset):
        return jnp.diag(v, k=offset)

    return allocdiag


@jax_funcify.register(AllocEmpty)
def jax_funcify_AllocEmpty(op, **kwargs):
    def allocempty(*shape):
        return jnp.empty(shape, dtype=op.dtype)

    return allocempty


@jax_funcify.register(Alloc)
def jax_funcify_Alloc(op, **kwargs):
    def alloc(x, *shape):
        res = jnp.broadcast_to(x, shape)
        return res

    return alloc


@jax_funcify.register(ARange)
def jax_funcify_ARange(op, node, **kwargs):
    """Register a JAX implementation for `ARange`.

    `jax.numpy.arange` requires concrete values for its arguments. Here we check
    that the arguments are constant, and raise otherwise.

    TODO: Handle other situations in which values are concrete (shape of an array).

    """
    arange_args = node.inputs
    constant_args = []
    for arg in arange_args:
        if not isinstance(arg, Constant):
            raise NotImplementedError(ARANGE_CONCRETE_VALUE_ERROR)

        constant_args.append(arg.value)

    start, stop, step = constant_args

    def arange(*_):
        return jnp.arange(start, stop, step, dtype=op.dtype)

    return arange


@jax_funcify.register(Join)
def jax_funcify_Join(op, **kwargs):
    def join(axis, *tensors):
        # tensors could also be tuples, and in this case they don't have a ndim
        tensors = [jnp.asarray(tensor) for tensor in tensors]
        view = op.view
        if (view != -1) and all(
            tensor.shape[axis] == 0 for tensor in tensors[0:view] + tensors[view + 1 :]
        ):
            return tensors[view]

        else:
            return jnp.concatenate(tensors, axis=axis)

    return join


@jax_funcify.register(ExtractDiag)
def jax_funcify_ExtractDiag(op, **kwargs):
    offset = op.offset
    axis1 = op.axis1
    axis2 = op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    return extract_diag


@jax_funcify.register(Eye)
def jax_funcify_Eye(op, **kwargs):
    dtype = op.dtype

    def eye(N, M, k):
        return jnp.eye(N, M, k, dtype=dtype)

    return eye


@jax_funcify.register(MakeVector)
def jax_funcify_MakeVector(op, **kwargs):
    def makevector(*x):
        return jnp.array(x, dtype=op.dtype)

    return makevector


@jax_funcify.register(TensorFromScalar)
def jax_funcify_TensorFromScalar(op, **kwargs):
    def tensor_from_scalar(x):
        return x

    return tensor_from_scalar


@jax_funcify.register(ScalarFromTensor)
def jax_funcify_ScalarFromTensor(op, **kwargs):
    def scalar_from_tensor(x):
        return jnp.array(x).flatten()[0]

    return scalar_from_tensor

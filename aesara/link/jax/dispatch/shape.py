import jax.numpy as jnp

from aesara.graph import Constant
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, Unbroadcast
from aesara.tensor.type import TensorType


class JAXShapeTuple(Op):
    """Dummy Op that represents a `size` specified as a tuple."""

    def make_node(self, *inputs):
        dtype = inputs[0].type.dtype
        otype = TensorType(dtype, shape=(len(inputs),))
        return Apply(self, inputs, [otype()])

    def perform(self, *inputs):
        return tuple(inputs)


@jax_funcify.register(JAXShapeTuple)
def jax_funcify_JAXShapeTuple(op, **kwargs):
    def shape_tuple_fn(*x):
        return tuple(x)

    return shape_tuple_fn


SHAPE_NOT_COMPATIBLE = """JAX requires concrete values for the `shape` parameter of `jax.numpy.reshape`.
Concrete values are either constants:

>>> import aesara.tensor as at
>>> x = at.ones(6)
>>> y = x.reshape((2, 3))

Or the shape of an array:

>>> mat = at.matrix('mat')
>>> y = x.reshape(mat.shape)
"""


def assert_shape_argument_jax_compatible(shape):
    """Assert whether the current node can be JIT-compiled by JAX.

    JAX can JIT-compile functions with a `shape` or `size` argument if it is
    given a concrete value, i.e. either a constant or the shape of any traced
    value.

    """
    shape_op = shape.owner.op
    if not isinstance(shape_op, (Shape, Shape_i, JAXShapeTuple)):
        raise NotImplementedError(SHAPE_NOT_COMPATIBLE)


@jax_funcify.register(Reshape)
def jax_funcify_Reshape(op, node, **kwargs):

    shape = node.inputs[1]

    if isinstance(shape, Constant):
        constant_shape = shape.data

        def reshape(x, shape):
            return jnp.reshape(x, constant_shape)

    else:
        assert_shape_argument_jax_compatible(shape)

        def reshape(x, shape):
            return jnp.reshape(x, shape)

    return reshape


@jax_funcify.register(Shape)
def jax_funcify_Shape(op, **kwargs):
    def shape(x):
        return jnp.shape(x)

    return shape


@jax_funcify.register(Shape_i)
def jax_funcify_Shape_i(op, **kwargs):
    i = op.i

    def shape_i(x):
        return jnp.shape(x)[i]

    return shape_i


@jax_funcify.register(SpecifyShape)
def jax_funcify_SpecifyShape(op, node, **kwargs):
    def specifyshape(x, *shape):
        assert x.ndim == len(shape)
        assert x.shape == tuple(shape), (
            "got shape",
            x.shape,
            "expected",
            shape,
        )
        return x

    return specifyshape


@jax_funcify.register(Unbroadcast)
def jax_funcify_Unbroadcast(op, **kwargs):
    def unbroadcast(x):
        return x

    return unbroadcast

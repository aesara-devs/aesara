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


@jax_funcify.register(Reshape)
def jax_funcify_Reshape(op, node, **kwargs):

    # JAX reshape only works with constant inputs, otherwise JIT fails
    shape = node.inputs[1]
    if isinstance(shape, Constant):
        constant_shape = shape.data

        def reshape(x, shape):
            return jnp.reshape(x, constant_shape)

    else:

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
def jax_funcify_SpecifyShape(op, **kwargs):
    def specifyshape(x, *shape):
        assert x.ndim == len(shape)
        assert jnp.all(x.shape == tuple(shape)), (
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

import jax.numpy as jnp

from aesara.graph import Constant
from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, Unbroadcast


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

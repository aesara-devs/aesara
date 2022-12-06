from aesara.compile import optdb
from aesara.graph.rewriting.basic import in2out, node_rewriter
from aesara.tensor.basic import MakeVector
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.op import RandomVariable


@node_rewriter([RandomVariable])
def size_parameter_as_tuple(fgraph, node):
    """Replace `MakeVector` and `DimShuffle` (when used to transform a scalar
    into a 1d vector) when they are found as the input of a `size` or `shape`
    parameter by `JAXShapeTuple` during transpilation.

    The JAX implementations of `MakeVector` and `DimShuffle` always return JAX
    `TracedArrays`, but JAX only accepts concrete values as inputs for the `size`
    or `shape` parameter. When these `Op`s are used to convert scalar or tuple
    inputs, however, we can avoid tracing by making them return a tuple of their
    inputs instead.

    Note that JAX does not accept scalar inputs for the `size` or `shape`
    parameters, and this rewrite also ensures that scalar inputs are turned into
    tuples during transpilation.

    """
    from aesara.link.jax.dispatch.shape import JAXShapeTuple

    size_arg = node.inputs[1]
    size_node = size_arg.owner

    if size_node is None:
        return

    if isinstance(size_node.op, JAXShapeTuple):
        return

    if isinstance(size_node.op, MakeVector) or (
        isinstance(size_node.op, DimShuffle)
        and size_node.op.input_broadcastable == ()
        and size_node.op.new_order == ("x",)
    ):
        # Here Aesara converted a tuple or list to a tensor
        new_size_args = JAXShapeTuple()(*size_node.inputs)
        new_inputs = list(node.inputs)
        new_inputs[1] = new_size_args

        new_node = node.clone_with_new_inputs(new_inputs)
        return new_node.outputs


optdb.register(
    "jax_size_parameter_as_tuple", in2out(size_parameter_as_tuple), "jax", position=100
)

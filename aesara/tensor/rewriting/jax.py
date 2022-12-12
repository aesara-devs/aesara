from aesara.compile import optdb
from aesara.graph.rewriting.basic import in2out, node_rewriter
from aesara.tensor.var import TensorVariable
import aesara.tensor as at
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedSubtensor
from aesara.tensor.math import Sum


@node_rewriter([AdvancedIncSubtensor])
def boolean_indexing_set_or_inc(fgraph, node):
    """Replace `AdvancedIncSubtensor` when using boolean indexing using `Switch`.

    JAX cannot JIT-compile functions that use boolean indexing to set values in
    an array. A workaround is to re-express this logic using `jax.numpy.where`.
    This rewrite allows to improve upon JAX's API.

    """

    op = node.op
    x = node.inputs[0]
    y = node.inputs[1]
    cond = node.inputs[2]

    if not isinstance(cond, TensorVariable):
        return

    if not cond.type.dtype == 'bool':
        return

    if op.set_instead_of_inc:
        out = at.where(cond, y, x)
        return out.owner.outputs
    else:
        out = at.where(cond, x + y, x)
        return out.owner.outputs


optdb.register(
    "jax_boolean_indexing_set_or_inc", in2out(boolean_indexing_set_or_inc), "jax", position=100
)


@node_rewriter([Sum])
def boolean_indexing_sum(fgraph, node):
    """Replace the sum of `AdvancedSubtensor` with boolean indexing.

    JAX cannot JIT-compile functions that use boolean indexing, but can compile
    those expressions that can be re-expressed using `jax.numpy.where`. This
    rewrite re-rexpressed the model on the behalf of the user and thus allows to
    improve upon JAX's API.

    """
    operand = node.inputs[0]

    if not isinstance(operand, TensorVariable):
        return

    if operand.owner is None:
        return

    if not isinstance(operand.owner.op, AdvancedSubtensor):
        return

    x = operand.owner.inputs[0]
    cond = operand.owner.inputs[1]

    if not isinstance(cond, TensorVariable):
        return

    if not cond.type.dtype == 'bool':
        return

    out = at.sum(at.where(cond, x, 0))
    return out.owner.outputs

optdb.register(
    "jax_boolean_indexing_sum", in2out(boolean_indexing_sum), "jax", position=100
)

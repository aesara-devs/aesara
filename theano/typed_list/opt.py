from theano.compile import optdb
from theano.graph.opt import TopoOptimizer, local_optimizer
from theano.typed_list.basic import Append, Extend, Insert, Remove, Reverse


@local_optimizer([Append, Extend, Insert, Reverse, Remove], inplace=True)
def typed_list_inplace_opt(fgraph, node):
    if (
        isinstance(node.op, (Append, Extend, Insert, Reverse, Remove))
        and not node.op.inplace
    ):

        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False


optdb.register(
    "typed_list_inplace_opt",
    TopoOptimizer(typed_list_inplace_opt, failure_callback=TopoOptimizer.warn_inplace),
    60,
    "fast_run",
    "inplace",
)

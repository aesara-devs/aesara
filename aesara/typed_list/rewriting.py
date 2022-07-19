from aesara.compile import optdb
from aesara.graph.rewriting.basic import WalkingGraphRewriter, node_rewriter
from aesara.typed_list.basic import Append, Extend, Insert, Remove, Reverse


@node_rewriter([Append, Extend, Insert, Reverse, Remove], inplace=True)
def typed_list_inplace_rewrite(fgraph, node):
    if (
        isinstance(node.op, (Append, Extend, Insert, Reverse, Remove))
        and not node.op.inplace
    ):

        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False


optdb.register(
    "typed_list_inplace_rewrite",
    WalkingGraphRewriter(
        typed_list_inplace_rewrite, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=60,
)

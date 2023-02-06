import warnings
from typing import TYPE_CHECKING, Generator, Optional, Sequence, Union, cast

from aesara.graph.basic import Apply, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.db import RewriteDatabaseQuery


if TYPE_CHECKING:
    from aesara.graph.rewriting.basic import GraphRewriter


def rewrite_graph(
    graph: Union[Variable, Sequence[Variable], FunctionGraph],
    include: Sequence[str] = ("canonicalize",),
    custom_rewrite: Optional["GraphRewriter"] = None,
    clone: bool = False,
    custom_opt: Optional["GraphRewriter"] = None,
    **kwargs,
) -> Union[Variable, Sequence[Variable], FunctionGraph]:
    """Easily apply rewrites to a graph.

    Parameters
    ----------
    graph
        A `FunctionGraph` or `Variable` to be rewritten.
    include
        String names of the rewrites to be queried, via a
        `RewriteDatabaseQuery` instance, and applied.  The default rewrite
        query string is ``"canonicalization"``.
    custom_rewrite
        A custom `Rewriter` to also be applied.
    clone
        Whether or not to clone the input graph before rewriting.
    **kwargs
        Keyword arguments passed to a `RewriteDatabaseQuery` object.
    """
    from aesara.compile import optdb

    return_fgraph = False
    if isinstance(graph, FunctionGraph):
        outputs: Sequence[Variable] = graph.outputs
        fgraph = graph
        return_fgraph = True
    else:
        if isinstance(graph, (list, tuple)):
            outputs = graph
        else:
            assert isinstance(graph, Variable)
            outputs = [graph]

        fgraph = FunctionGraph(outputs=outputs, clone=clone)

    query_rewrites = optdb.query(RewriteDatabaseQuery(include=include, **kwargs))
    _ = query_rewrites.rewrite(fgraph)

    if custom_opt is not None:
        warnings.warn(
            "`custom_opt` is deprecated; use `custom_rewrite` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        custom_rewrite = custom_opt

    if custom_rewrite:
        custom_rewrite.rewrite(fgraph)

    if return_fgraph:
        return fgraph
    else:
        if isinstance(graph, (list, tuple)):
            return fgraph.outputs
        else:
            return fgraph.outputs[0]


def get_clients_at_depth(
    fgraph: FunctionGraph, node: Apply, depth: int
) -> Generator[Apply, None, None]:
    """Yields node clients at given depth."""
    for var in node.outputs:
        if depth > 0:
            for out_node, _ in fgraph.clients[var]:
                if out_node == "output":
                    continue
                yield from get_clients_at_depth(
                    fgraph, cast(Apply, out_node), depth - 1
                )
        else:
            assert var.owner is not None
            yield var.owner


DEPRECATED_NAMES = [
    (
        "optimize_graph",
        "`optimize_graph` is deprecated: use `rewrite_graph` instead.",
        rewrite_graph,
    ),
]


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")

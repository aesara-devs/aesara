import sys

import pytest

from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import graph_rewriter
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.type import vectors


def test_rewrite_graph():

    x, y = vectors("xy")

    @graph_rewriter
    def custom_rewrite(fgraph):
        fgraph.replace(x, y, import_missing=True)

    x_rewritten = rewrite_graph(x, custom_rewrite=custom_rewrite)

    assert x_rewritten is y

    x_rewritten = rewrite_graph(
        FunctionGraph(outputs=[x], clone=False), custom_rewrite=custom_rewrite
    )

    assert x_rewritten.outputs[0] is y


def test_deprecations():
    """Make sure we can import deprecated classes from current and deprecated modules."""
    with pytest.deprecated_call():
        from aesara.graph.rewriting.utils import optimize_graph  # noqa: F401 F811

    with pytest.deprecated_call():
        from aesara.graph.opt_utils import optimize_graph  # noqa: F401 F811

    del sys.modules["aesara.graph.opt_utils"]

    with pytest.deprecated_call():
        from aesara.graph.opt_utils import rewrite_graph  # noqa: F401

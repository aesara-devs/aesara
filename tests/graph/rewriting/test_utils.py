import sys

import pytest

from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import graph_rewriter
from aesara.graph.rewriting.utils import is_same_graph, rewrite_graph
from aesara.tensor.math import neg
from aesara.tensor.type import vectors


class TestIsSameGraph:
    def check(self, expected):
        """
        Core function to perform comparison.

        :param expected: A list of tuples (v1, v2, ((g1, o1), ..., (gN, oN)))
        with:
            - `v1` and `v2` two Variables (the graphs to be compared)
            - `gj` a `givens` dictionary to give as input to `is_same_graph`
            - `oj` the expected output of `is_same_graph(v1, v2, givens=gj)`

        This function also tries to call `is_same_graph` by inverting `v1` and
        `v2`, and ensures the output remains the same.
        """
        for v1, v2, go in expected:
            for gj, oj in go:
                r1 = is_same_graph(v1, v2, givens=gj)
                assert r1 == oj
                r2 = is_same_graph(v2, v1, givens=gj)
                assert r2 == oj

    def test_single_var(self):
        # Test `is_same_graph` with some trivial graphs (one Variable).

        x, y, z = vectors("x", "y", "z")
        self.check(
            [
                (x, x, (({}, True),)),
                (
                    x,
                    y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (x, neg(x), (({}, False),)),
                (x, neg(y), (({}, False),)),
            ]
        )

    def test_full_graph(self):
        # Test `is_same_graph` with more complex graphs.

        x, y, z = vectors("x", "y", "z")
        t = x * y
        self.check(
            [
                (x * 2, x * 2, (({}, True),)),
                (
                    x * 2,
                    y * 2,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * 2,
                    y * 2,
                    (
                        ({}, False),
                        ({x: y}, True),
                    ),
                ),
                (
                    x * 2,
                    y * 3,
                    (
                        ({}, False),
                        ({y: x}, False),
                    ),
                ),
                (
                    t * 2,
                    z * 2,
                    (
                        ({}, False),
                        ({t: z}, True),
                    ),
                ),
                (
                    t * 2,
                    z * 2,
                    (
                        ({}, False),
                        ({z: t}, True),
                    ),
                ),
                (x * (y * z), (x * y) * z, (({}, False),)),
            ]
        )

    def test_merge_only(self):
        # Test `is_same_graph` when `equal_computations` cannot be used.

        x, y, z = vectors("x", "y", "z")
        t = x * y
        self.check(
            [
                (x, t, (({}, False), ({t: x}, True))),
                (
                    t * 2,
                    x * 2,
                    (
                        ({}, False),
                        ({t: x}, True),
                    ),
                ),
                (
                    x * x,
                    x * y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * x,
                    x * y,
                    (
                        ({}, False),
                        ({y: x}, True),
                    ),
                ),
                (
                    x * x + z,
                    x * y + t,
                    (({}, False), ({y: x}, False), ({y: x, t: z}, True)),
                ),
            ],
        )


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

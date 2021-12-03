from typing import Callable, Iterator, List, Union

from etuples.core import ExpressionTuple
from kanren import run
from unification import var
from unification.variable import Var

from aesara.graph.basic import Apply, Variable
from aesara.graph.opt import LocalOptimizer
from aesara.graph.unify import eval_if_etuple


class KanrenRelationSub(LocalOptimizer):
    r"""A local optimizer that uses `kanren` to match and replace terms.

    See `kanren <https://github.com/pythological/kanren>`__ for more information
    miniKanren and the API for constructing `kanren` goals.

    Example
    -------

    ..code-block:: python

        from kanren import eq, conso, var

        import aesara.tensor as at
        from aesara.graph.kanren import KanrenRelationSub


        def relation(in_lv, out_lv):
            # A `kanren` goal that changes `at.log` terms to `at.exp`
            cdr_lv = var()
            return eq(conso(at.log, cdr_lv, in_lv),
                    conso(at.exp, cdr_lv, out_lv))


        kanren_sub_opt = KanrenRelationSub(relation)

    """

    reentrant = True

    def __init__(
        self,
        kanren_relation: Callable[[Variable, Var], Callable],
        results_filter: Callable[
            [Iterator], List[Union[ExpressionTuple, Variable]]
        ] = lambda x: next(x, None),
        node_filter: Callable[[Apply], bool] = lambda x: True,
    ):
        r"""Create a `KanrenRelationSub`.

        Parameters
        ----------
        kanren_relation
            A function that takes an input graph and an output logic variable and
            returns a `kanren` goal.
        results_filter
            A function that takes the direct output of `kanren.run(None, ...)`
            and returns a single result.  The default implementation returns
            the first result.
        node_filter
            A function taking a single node and returns ``True`` when the node
            should be processed.
        """
        self.kanren_relation = kanren_relation
        self.results_filter = results_filter
        self.node_filter = node_filter
        super().__init__()

    def transform(self, fgraph, node):
        if self.node_filter(node) is False:
            return False

        try:
            input_expr = node.default_output()
        except ValueError:
            input_expr = node.outputs

        q = var()
        kanren_results = run(None, q, self.kanren_relation(input_expr, q))

        chosen_res = self.results_filter(kanren_results)

        if chosen_res:
            if isinstance(chosen_res, list):
                new_outputs = [eval_if_etuple(v) for v in chosen_res]
            else:
                new_outputs = [eval_if_etuple(chosen_res)]

            return new_outputs
        else:
            return False

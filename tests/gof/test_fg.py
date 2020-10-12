import os
import pickle

import pytest

import theano
from theano.compat import PY3
from theano.gof.fg import FunctionGraph
from theano import tensor as tt


class TestFunctionGraph:
    def test_pickle(self):
        v = tt.vector()
        func = FunctionGraph([v], [v + 1])

        s = pickle.dumps(func)
        pickle.loads(s)

    @pytest.mark.skipif(
        not theano.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    @pytest.mark.slow
    def test_node_outputs_not_used(self):
        # In the past, we where removing some not used variable from
        # fgraph.variables event if the apply had other output used in
        # the graph. This caused a crash.
        # This test run the pickle that reproduce this case.
        with open(
            os.path.join(os.path.dirname(__file__), "test_fg_old_crash.pkl"), "rb"
        ) as f:
            from theano.misc.pkl_utils import CompatUnpickler

            if PY3:
                u = CompatUnpickler(f, encoding="latin1")
            else:
                u = CompatUnpickler(f)
            d = u.load()
        f = theano.function(**d)

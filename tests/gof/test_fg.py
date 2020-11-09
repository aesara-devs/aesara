import pickle

from aesara import tensor as tt
from aesara.gof.fg import FunctionGraph


class TestFunctionGraph:
    def test_pickle(self):
        v = tt.vector()
        func = FunctionGraph([v], [v + 1])

        s = pickle.dumps(func)
        pickle.loads(s)

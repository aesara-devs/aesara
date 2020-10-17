import pickle

from theano import tensor as tt
from theano.gof.fg import FunctionGraph


class TestFunctionGraph:
    def test_pickle(self):
        v = tt.vector()
        func = FunctionGraph([v], [v + 1])

        s = pickle.dumps(func)
        pickle.loads(s)

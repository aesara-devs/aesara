import theano
from theano.gof.utils import remove


def test_remove():
    def even(x):
        return x % 2 == 0

    def odd(x):
        return x % 2 == 1

    # The list are needed as with python 3, remove and filter return generators
    # and we can't compare generators.
    assert list(remove(even, range(5))) == list(filter(odd, range(5)))


def test_stack_trace():
    orig = theano.config.traceback__limit
    try:
        theano.config.traceback__limit = 1
        v = theano.tensor.vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 1
        theano.config.traceback__limit = 2
        v = theano.tensor.vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 2
    finally:
        theano.config.traceback__limit = orig

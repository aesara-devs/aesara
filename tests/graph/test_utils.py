import theano


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

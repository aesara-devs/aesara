import aesara
from aesara.tensor.type import vector


def test_stack_trace():
    with aesara.config.change_flags(traceback__limit=1):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 1

    with aesara.config.change_flags(traceback__limit=2):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 2

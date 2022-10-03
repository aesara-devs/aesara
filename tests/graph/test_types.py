from typing import Any

import pytest

from aesara.graph.basic import Variable
from aesara.graph.type import NewTypeMeta, Props, Type
from aesara.issubtype import issubtype


class MyTypeMeta(NewTypeMeta):
    thingy: Props[Any] = None

    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, MyTypeMeta) and other.thingy == self.thingy

    def __hash__(self):
        return hash((MyTypeMeta, self.thingy))

    def __str__(self):
        return f"R{self.thingy}"

    def __repr__(self):
        return f"R{self.thingy}"


class MyType(Type, metaclass=MyTypeMeta):
    pass


class MyTypeMeta2(MyTypeMeta):
    def is_super(self, other):
        if self.thingy <= other.thingy:
            return True


class MyType2(Type, metaclass=MyTypeMeta2):
    pass


def test_is_super():
    t1 = MyType.subtype(1)
    t2 = MyType.subtype(2)

    assert t1.is_super(t2) is None

    t1_2 = MyType.subtype(1)
    assert t1.is_super(t1_2)


def test_in_same_class():
    t1 = MyType.subtype(1)
    t2 = MyType.subtype(2)

    assert t1.in_same_class(t2) is False

    t1_2 = MyType.subtype(1)
    assert t1.in_same_class(t1_2)


def test_convert_variable():
    t1 = MyType.subtype(1)
    v1 = Variable(MyType.subtype(1), None, None)
    v2 = Variable(MyType.subtype(2), None, None)
    v3 = Variable(MyType2.subtype(0), None, None)

    assert t1.convert_variable(v1) is v1
    assert t1.convert_variable(v2) is None

    with pytest.raises(NotImplementedError):
        t1.convert_variable(v3)


def test_default_clone():
    mt = MyType.subtype(1)
    assert issubtype(mt.clone(1), MyType)

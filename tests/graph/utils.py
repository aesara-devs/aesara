import numpy as np

from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.graph.type import Type


def is_variable(x):
    if not isinstance(x, Variable):
        raise TypeError("not a Variable", x)
    return x


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


class MyType2(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


def MyVariable(name):
    return Variable(MyType(), None, None, name=name)


def MyVariable2(name):
    return Variable(MyType2(), None, None, name=name)


class MyOp(Op):
    def __init__(self, name, dmap=None, x=None):
        self.name = name
        if dmap is None:
            dmap = {}
        self.destroy_map = dmap
        self.x = x

    def make_node(self, *inputs):
        inputs = list(map(is_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType()()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        outputs[0] = np.array(inputs, dtype=np.object)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        # rval = (self is other) or (isinstance(other, MyOp) and self.x is not None and self.x == other.x and self.name == other.name)
        rval = (self is other) or (
            isinstance(other, MyOp) and self.x is not None and self.x == other.x
        )
        return rval

    def __hash__(self):
        # return hash(self.x if self.x is not None else id(self)) ^ hash(self.name)
        if self.x is not None:
            return hash(self.x)
        else:
            return id(self)


op1 = MyOp("Op1")
op2 = MyOp("Op2")
op3 = MyOp("Op3")
op4 = MyOp("Op4")
op5 = MyOp("Op5")
op6 = MyOp("Op6")
op_d = MyOp("OpD", {0: [0]})

op_y = MyOp("OpY", x=1)
op_z = MyOp("OpZ", x=1)

import numpy as np

import theano.tensor.basic as tt
from theano.gof.fg import FunctionGraph
from theano.gof.graph import Apply, Variable
from theano.gof.op import Op
from theano.gof.opt import MergeOptimizer
from theano.gof.type import Type


def is_variable(x):
    if not isinstance(x, Variable):
        raise TypeError("not a Variable", x)
    return x


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)


class MyOp(Op):
    def __init__(self, name, dmap=None, x=None):
        if dmap is None:
            dmap = {}
        self.name = name
        self.destroy_map = dmap
        self.x = x

    def make_node(self, *inputs):
        inputs = list(map(is_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType()()]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return (
            self is other
            or isinstance(other, MyOp)
            and self.x is not None
            and self.x == other.x
        )

    def __hash__(self):
        if self.x is not None:
            return self.x
        else:
            return id(self)


op1 = MyOp("Op1")


def test_merge_with_weird_eq():
    # numpy arrays don't compare equal like other python objects

    # SCALAR CASE
    x = tt.constant(np.asarray(1), name="x")
    y = tt.constant(np.asarray(1), name="y")
    g = FunctionGraph([x, y], [x + y])
    MergeOptimizer().optimize(g)

    assert len(g.apply_nodes) == 1
    node = list(g.apply_nodes)[0]
    assert len(node.inputs) == 2
    assert node.inputs[0] is node.inputs[1]

    # NONSCALAR CASE
    # This was created to test TensorConstantSignature
    x = tt.constant(np.ones(5), name="x")
    y = tt.constant(np.ones(5), name="y")
    g = FunctionGraph([x, y], [x + y])
    MergeOptimizer().optimize(g)

    assert len(g.apply_nodes) == 1
    node = list(g.apply_nodes)[0]
    assert len(node.inputs) == 2
    assert node.inputs[0] is node.inputs[1]

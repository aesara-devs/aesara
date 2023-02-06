import numpy as np

from aesara.graph.basic import Apply, Constant, NominalVariable, Variable, clone_replace
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import HasInnerGraph, Op
from aesara.graph.type import Type


def is_variable(x):
    if not isinstance(x, Variable):
        raise TypeError(f"not a Variable: {x}")
    return x


class MyType(Type):
    def filter(self, data, **kwargs):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)

    def __repr__(self):
        return "MyType()"


class MyType2(Type):
    def filter(self, data, **kwargs):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType2)

    def __hash__(self):
        return hash(MyType)

    def __repr__(self):
        return "MyType2()"


def MyVariable(name):
    return Variable(MyType(), None, None, name=name)


def MyConstant(name, data=None):
    return Constant(MyType(), data, name=name)


def MyVariable2(name):
    return Variable(MyType2(), None, None, name=name)


class MyOp(Op):
    def __init__(self, name, dmap=None, x=None, n_outs=1):
        self.name = name
        if dmap is None:
            dmap = {}
        self.destroy_map = dmap
        self.x = x
        self.n_outs = n_outs

    def make_node(self, *inputs):
        inputs = list(map(is_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType()() for i in range(self.n_outs)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        outputs[0] = np.array(inputs, dtype=object)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        rval = (self is other) or (
            isinstance(other, MyOp)
            and self.x is not None
            and self.x == other.x
            and self.n_outs == other.n_outs
        )
        return rval

    def __hash__(self):
        if self.x is not None:
            return hash((self.x, self.n_outs))
        else:
            return hash((id(self), self.n_outs))


class MyOpCastType2(MyOp):
    def make_node(self, *inputs):
        inputs = list(map(is_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")

        outputs = [MyType2()()]
        return Apply(self, inputs, outputs)


class MyOpMultipleOutputs(MyOp):
    def make_node(self, input):
        outputs = [input.type(), input.type()]
        return Apply(self, [input], outputs)


op1 = MyOp("Op1")
op2 = MyOp("Op2")
op3 = MyOp("Op3")
op4 = MyOp("Op4")
op5 = MyOp("Op5")
op6 = MyOp("Op6")
op_d = MyOp("OpD", {0: [0]})

op_y = MyOp("OpY", x=1)
op_z = MyOp("OpZ", x=1)

op_cast_type2 = MyOpCastType2("OpCastType2")
op_multiple_outputs = MyOpMultipleOutputs("OpMultipleOutputs")


class MyInnerGraphOp(Op, HasInnerGraph):
    __props__ = ()

    def __init__(self, inner_inputs, inner_outputs):
        input_replacements = [
            (v, NominalVariable(n, v.type))
            for n, v in enumerate(inner_inputs)
            if not isinstance(v, Constant)
        ]
        outputs = clone_replace(inner_outputs, replace=input_replacements)
        _, inputs = zip(*input_replacements) if input_replacements else (None, [])
        self.fgraph = FunctionGraph(inputs, outputs, clone=False)

    def make_node(self, *inputs):
        outputs = [inputs[0].type()]
        return Apply(self, list(inputs), outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")

    @property
    def fn(self):
        raise NotImplementedError("No Python implementation available.")

    @property
    def inner_inputs(self):
        return self.fgraph.inputs

    @property
    def inner_outputs(self):
        return self.fgraph.outputs

    def clone(self):
        return type(self)(self.fgraph.inputs, self.fgraph.outputs)

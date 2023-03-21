from copy import deepcopy
from typing import Callable

import numpy as np

import aesara
from aesara.compile.mode import Mode
from aesara.graph import fg
from aesara.graph.basic import Apply, Constant, Variable, clone
from aesara.graph.op import Op
from aesara.graph.type import Type
from aesara.link.basic import Container, Linker, PerformLinker, WrapLinker
from aesara.link.c.basic import OpWiseCLinker
from aesara.tensor.type import matrix, scalar
from aesara.utils import cmp, to_return_values


def make_function(linker: Linker, unpack_single: bool = True, **kwargs) -> Callable:
    """
    Returns a function that takes values corresponding to the inputs of the
    fgraph used by this L{Linker} and returns values corresponding the the
    outputs of that fgraph. If inplace is True, the calculations will
    operate in the same storage the fgraph uses, else independent storage
    will be allocated for the function.

    Parameters
    ----------
    unpack_single : bool
        If `unpack_single` is True (default) and that the function has only one
        output, then that output will be returned. Else, a list or tuple of
        length 1 will be returned.

    Examples
    --------
    e = x + y
    fgraph = FunctionGraph([x, y], [e])
    fn = make_function(MyLinker(fgraph), inplace)
    print fn(1.0, 2.0) # 3.0
    print e.data # 3.0 iff inplace == True (else unknown)

    """
    thunk, inputs, outputs = linker.make_thunk(**kwargs)

    def execute(*args):
        takes = len(inputs)
        got = len(args)
        if got != takes:
            raise TypeError(f"Function call takes exactly {takes} args ({got} given)")
        for arg, variable in zip(args, inputs):
            variable.data = arg
        thunk()
        if unpack_single:
            return to_return_values([variable.data for variable in outputs])
        else:
            return [variable.data for variable in outputs]

    return execute


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class TDouble(Type):
    def filter(self, data):
        return float(data)


tdouble = TDouble()


def double(name):
    return Variable(tdouble, None, None, name=name)


class MyOp(Op):
    __props__ = ("nin", "name", "impl")

    def __init__(self, nin, name, impl=None):
        self.nin = nin
        self.name = name
        if impl:
            self.impl = impl

    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = [as_variable(i) for i in inputs]
        for input in inputs:
            if input.type is not tdouble:
                raise Exception("Error 1")
        outputs = [double(self.name + "_R")]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def perform(self, node, inputs, out_):
        (out,) = out_
        out[0] = self.impl(*inputs)


add = MyOp(2, "Add", lambda x, y: x + y)
sub = MyOp(2, "Sub", lambda x, y: x - y)
mul = MyOp(2, "Mul", lambda x, y: x * y)
div = MyOp(2, "Div", lambda x, y: x / y)


def notimpl(self, x):
    raise NotImplementedError()


raise_err = MyOp(1, "RaiseErr", notimpl)


def inputs():
    x = double("x")
    y = double("y")
    z = double("z")
    return x, y, z


def perform_linker(fgraph):
    lnk = PerformLinker().accept(fgraph)
    return lnk


def FunctionGraph(inputs, outputs):
    e = fg.FunctionGraph(inputs, outputs)
    return e


class TestPerformLinker:
    def test_thunk(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = perform_linker(FunctionGraph([x, y, z], [e])).make_thunk()
        i[0].data = 1
        assert i[0].data == 1
        i[1].data = 2
        assert i[1].data == 2
        fn()
        assert o[0].data == 1.5

    def test_function(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn = make_function(perform_linker(FunctionGraph([x, y, z], [e])))
        assert fn(1.0, 2.0, 3.0) == 1.5

    def test_constant(self):
        x, y, z = inputs()
        y = Constant(tdouble, 2.0)
        e = mul(add(x, y), div(x, y))
        fn = make_function(perform_linker(FunctionGraph([x], [e])))
        assert fn(1.0) == 1.5

    def test_input_output_same(self):
        x, y, z = inputs()
        fn = make_function(perform_linker(FunctionGraph([x], [x])))
        assert 1.0 == fn(1.0)

    def test_input_dependency0(self):
        x, y, z = inputs()
        a, d = add(x, y), div(x, y)
        e = mul(a, d)
        fn = make_function(perform_linker(FunctionGraph(*clone([x, y, a], [e]))))
        assert fn(1.0, 2.0, 9.0) == 4.5

    def test_skiphole(self):
        x, y, z = inputs()
        a = add(x, y)
        r = raise_err(a)
        e = add(r, a)
        fn = make_function(perform_linker(FunctionGraph(*clone([x, y, r], [e]))))
        assert fn(1.0, 2.0, 4.5) == 7.5


def wrap_linker(fgraph, linkers, wrapper):
    lnk = WrapLinker(linkers, wrapper).accept(fgraph)
    return lnk


class TestWrapLinker:
    def test_0(self):
        nodes = []

        def wrap(fgraph, i, node, th):
            nodes.append(node.op)

        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = wrap_linker(
            FunctionGraph([x, y, z], [e]), [PerformLinker(allow_gc=False)], wrap
        ).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert nodes == [div, add, mul] or nodes == [add, div, mul]
        assert o[0].data is None

    def test_1(self):
        nodes = []

        def wrap(fgraph, i, node, th):
            nodes.append(node.op)
            th()

        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = wrap_linker(
            FunctionGraph([x, y, z], [e]), [PerformLinker(allow_gc=False)], wrap
        ).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert nodes == [div, add, mul] or nodes == [add, div, mul]
        assert o[0].data == 1.5


def test_sort_schedule_fn():
    from aesara.graph.sched import make_depends, sort_schedule_fn

    x = matrix("x")
    y = aesara.tensor.dot(x[:5] * 2, x.T + 1).T

    def str_cmp(a, b):
        return cmp(str(a), str(b))  # lexicographical sort

    linker = OpWiseCLinker(schedule=sort_schedule_fn(str_cmp))
    mode = Mode(linker=linker)
    f = aesara.function((x,), (y,), mode=mode)

    nodes = f.maker.linker.make_all()[-1]
    depends = make_depends()
    for a, b in zip(nodes[:-1], nodes[1:]):
        if not depends((b, a)):
            assert str(a) < str(b)


def test_container_deepcopy():
    # This is a test to a work around a NumPy bug.

    t = scalar()
    # It seam that numpy.asarray(0.).astype(floatX) can return a numpy
    # scalar with some NumPy Version. So we call numpy.asarray with
    # the dtype parameter.
    v = np.asarray(0.0, dtype=aesara.config.floatX)
    assert isinstance(v, np.ndarray), type(v)
    for readonly in [True, False]:
        c = Container(t, [v], readonly=readonly)
        assert isinstance(c.storage[0], np.ndarray), (c.storage[0], type(c.storage[0]))
        assert c.storage[0].dtype == v.dtype, (c.storage[0].dtype, v.dtype)
        assert c.storage[0].dtype == c.type.dtype, (c.storage[0].dtype, c.type.dtype)
        d = deepcopy(c)
        assert isinstance(d.storage[0], np.ndarray), (d.storage[0], type(d.storage[0]))
        assert d.storage[0].dtype == v.dtype, (d.storage[0].dtype, v.dtype)
        assert d.storage[0].dtype == c.type.dtype, (d.storage[0].dtype, c.type.dtype)

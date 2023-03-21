import numpy as np
import pytest

import aesara
import aesara.graph.op as op
import aesara.tensor as at
from aesara import shared
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.graph.type import Type
from aesara.graph.utils import TestValueError
from aesara.link.c.type import Generic
from aesara.tensor.math import log
from aesara.tensor.type import dmatrix, dscalar, dvector, vector


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):
    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return type(other) == type(self) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)

    def filter(self, x, strict=False, allow_downcast=None):
        # Dummy filter: we want this type to represent strings that
        # start with `self.thingy`.
        if not isinstance(x, str):
            raise TypeError("Invalid type")
        if not x.startswith(self.thingy):
            raise ValueError("Invalid value")
        return x

    # Added to make those tests pass in DebugMode
    @staticmethod
    def may_share_memory(a, b):
        # As this represent a string and string are immutable, they
        # never share memory in the DebugMode sense. This is needed as
        # Python reuse string internally.
        return False


class MyOp(Op):
    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
            outputs = [MyType(sum(input.type.thingy for input in inputs))()]
            return Apply(self, inputs, outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")


MyOp = MyOp()


class NoInputOp(Op):
    """An Op to test the corner-case of an Op with no input."""

    __props__ = ()

    def make_node(self):
        return Apply(self, [], [MyType("test")()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = "test Op no input"


class TestOp:
    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyType(1)(), MyType(2)()
        node = MyOp.make_node(r1, r2)
        # Are the inputs what I provided?
        assert [x for x in node.inputs] == [r1, r2]
        # Are the outputs what I expect?
        assert [x.type for x in node.outputs] == [MyType(3)]
        assert node.outputs[0].owner is node and node.outputs[0].index == 0

    # validate
    def test_validate(self):
        try:
            MyOp(Generic()(), MyType(1)())  # MyOp requires MyType instances
            raise Exception("Expected an exception")
        except Exception as e:
            if str(e) != "Error 1":
                raise

    def test_op_no_input(self):
        x = NoInputOp()()
        f = aesara.function([], x)
        rval = f()
        assert rval == "test Op no input"


class TestMakeThunk:
    def test_no_make_node(self):
        class DoubleOp(Op):
            """An Op without make_node"""

            __props__ = ()

            itypes = [dmatrix]
            otypes = [dmatrix]

            def perform(self, node, inputs, outputs):
                inp = inputs[0]
                output = outputs[0]
                output[0] = inp * 2

        x_input = dmatrix("x_input")
        f = aesara.function([x_input], DoubleOp()(x_input))
        inp = np.random.random((5, 4))
        out = f(inp)
        assert np.allclose(inp * 2, out)


def test_test_value_python_objects():
    for x in ([0, 1, 2], 0, 0.5, 1):
        assert np.all(op.get_test_value(x) == x)


def test_test_value_ndarray():
    x = np.zeros((5, 5))
    v = op.get_test_value(x)
    assert np.all(v == x)


def test_test_value_constant():
    x = at.as_tensor_variable(np.zeros((5, 5)))
    v = op.get_test_value(x)

    assert np.all(v == np.zeros((5, 5)))


def test_test_value_shared():
    x = shared(np.zeros((5, 5)))
    v = op.get_test_value(x)

    assert np.all(v == np.zeros((5, 5)))


@config.change_flags(compute_test_value="raise")
def test_test_value_op():
    x = log(np.ones((5, 5)))
    v = op.get_test_value(x)

    assert np.allclose(v, np.zeros((5, 5)))


@config.change_flags(compute_test_value="off")
def test_get_test_values_no_debugger():
    """Tests that `get_test_values` returns `[]` when debugger is off."""

    x = vector()
    assert op.get_test_values(x) == []


@config.change_flags(compute_test_value="ignore")
def test_get_test_values_ignore():
    """Tests that `get_test_values` returns `[]` when debugger is set to "ignore" and some values are missing."""

    x = vector()
    assert op.get_test_values(x) == []


def test_get_test_values_success():
    """Tests that `get_test_values` returns values when available (and the debugger is on)."""

    for mode in ["ignore", "warn", "raise"]:
        with config.change_flags(compute_test_value=mode):
            x = vector()
            x.tag.test_value = np.zeros((4,), dtype=config.floatX)
            y = np.zeros((5, 5))

            iters = 0

            for x_val, y_val in op.get_test_values(x, y):
                assert x_val.shape == (4,)
                assert y_val.shape == (5, 5)

                iters += 1

            assert iters == 1


@config.change_flags(compute_test_value="raise")
def test_get_test_values_exc():
    """Tests that `get_test_values` raises an exception when debugger is set to raise and a value is missing."""

    with pytest.raises(TestValueError):
        x = vector()
        assert op.get_test_values(x) == []


def test_op_invalid_input_types():
    class TestOp(aesara.graph.op.Op):
        itypes = [dvector, dvector, dvector]
        otypes = [dvector]

        def perform(self, node, inputs, outputs):
            pass

    msg = r"^Invalid input types for Op.*"
    with pytest.raises(TypeError, match=msg):
        TestOp()(dvector(), dscalar(), dvector())


def test_op_input_broadcastable():
    # Test that we can create an op with a broadcastable subtype as input
    class SomeOp(aesara.tensor.Op):
        itypes = [at.dvector]
        otypes = [at.dvector]

        def perform(self, *_):
            raise NotImplementedError()

    x = at.TensorType(dtype="float64", shape=(1,))("x")
    assert SomeOp()(x).type == at.dvector

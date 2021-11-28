import numpy as np
import pytest

import aesara
import aesara.graph.op as op
import aesara.tensor as aet
from aesara import scalar as aes
from aesara import shared
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import COp, Op
from aesara.graph.type import Generic, Type
from aesara.graph.utils import MethodNotDefined, TestValueError
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
            outputs = [MyType(sum([input.type.thingy for input in inputs]))()]
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


class StructOp(COp):
    __props__ = ()

    def do_constant_folding(self, fgraph, node):
        # we are not constant
        return False

    # The input only serves to distinguish thunks
    def make_node(self, i):
        return Apply(self, [i], [aes.uint64()])

    def c_support_code_struct(self, node, name):
        return f"npy_uint64 counter{name};"

    def c_init_code_struct(self, node, name, sub):
        return f"counter{name} = 0;"

    def c_code(self, node, name, input_names, outputs_names, sub):
        return """
%(out)s = counter%(name)s;
counter%(name)s++;
""" % dict(
            out=outputs_names[0], name=name
        )

    def c_code_cache_version(self):
        return (1,)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")


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

    @pytest.mark.skipif(
        not config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_op_struct(self):
        sop = StructOp()
        c = sop(aesara.tensor.constant(0))
        mode = None
        if config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        f = aesara.function([], c, mode=mode)
        rval = f()
        assert rval == 0
        rval = f()
        assert rval == 1

        c2 = sop(aesara.tensor.constant(1))
        f2 = aesara.function([], [c, c2], mode=mode)
        rval = f2()
        assert rval == [0, 0]


class TestMakeThunk:
    def test_no_c_code(self):
        class IncOnePython(COp):
            """An Op with only a Python (perform) implementation"""

            __props__ = ()

            def make_node(self, input):
                input = aes.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def perform(self, node, inputs, outputs):
                (input,) = inputs
                (output,) = outputs
                output[0] = input + 1

        i = aes.int32("i")
        o = IncOnePython()(i)

        # Check that the c_code function is not implemented
        with pytest.raises(NotImplementedError):
            o.owner.op.c_code(o.owner, "o", ["x"], "z", {"fail": ""})

        storage_map = {i: [np.int32(3)], o: [None]}
        compute_map = {i: [True], o: [False]}

        thunk = o.owner.op.make_thunk(
            o.owner, storage_map, compute_map, no_recycling=[]
        )

        required = thunk()
        # Check everything went OK
        assert not required  # We provided all inputs
        assert compute_map[o][0]
        assert storage_map[o][0] == 4

    def test_no_perform(self):
        class IncOneC(COp):
            """An Op with only a C (c_code) implementation"""

            __props__ = ()

            def make_node(self, input):
                input = aes.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def c_code(self, node, name, inputs, outputs, sub):
                (x,) = inputs
                (z,) = outputs
                return f"{z} = {x} + 1;"

            def perform(self, *args, **kwargs):
                raise NotImplementedError("No Python implementation available.")

        i = aes.int32("i")
        o = IncOneC()(i)

        # Check that the perform function is not implemented
        with pytest.raises((NotImplementedError, MethodNotDefined)):
            o.owner.op.perform(o.owner, 0, [None])

        storage_map = {i: [np.int32(3)], o: [None]}
        compute_map = {i: [True], o: [False]}

        thunk = o.owner.op.make_thunk(
            o.owner, storage_map, compute_map, no_recycling=[]
        )
        if config.cxx:
            required = thunk()
            # Check everything went OK
            assert not required  # We provided all inputs
            assert compute_map[o][0]
            assert storage_map[o][0] == 4
        else:
            with pytest.raises((NotImplementedError, MethodNotDefined)):
                thunk()

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
        inp = np.random.rand(5, 4)
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
    x = aet.as_tensor_variable(np.zeros((5, 5)))
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

    msg = r"^Invalid input types for Op TestOp:\nInput 2/3: Expected TensorType\(float64, vector\)"
    with pytest.raises(TypeError, match=msg):
        TestOp()(dvector(), dscalar(), dvector())

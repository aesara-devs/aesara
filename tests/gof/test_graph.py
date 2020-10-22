import pickle
from itertools import count

import numpy as np
import pytest

from theano import shared, sparse, tensor
from theano.gof.graph import (
    Apply,
    Variable,
    as_string,
    clone,
    equal_computations,
    general_toposort,
    inputs,
    io_toposort,
)
from theano.gof.op import Op
from theano.gof.type import Type


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):
    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return isinstance(other, MyType) and other.thingy == self.thingy

    def __str__(self):
        return "R%s" % str(self.thingy)

    def __repr__(self):
        return "R%s" % str(self.thingy)


def MyVariable(thingy):
    return Variable(MyType(thingy), None, None)


class MyOp(Op):

    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                print(input, input.type, type(input), type(input.type))
                raise Exception("Error 1")
        outputs = [MyVariable(sum([input.type.thingy for input in inputs]))]
        return Apply(self, inputs, outputs)


MyOp = MyOp()


class TestInputs:
    def test_inputs(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        assert inputs(node.outputs) == [r1, r2]

    def test_inputs_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        i = inputs(node2.outputs)
        assert i == [r1, r2, r5], i


class X:
    def leaf_formatter(self, leaf):
        return str(leaf.type)

    def node_formatter(self, node, argstrings):
        return "{}({})".format(node.op, ", ".join(argstrings))

    def str(self, inputs, outputs):
        return as_string(
            inputs,
            outputs,
            leaf_formatter=self.leaf_formatter,
            node_formatter=self.node_formatter,
        )


class TestStr(X):
    def test_as_string(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        s = self.str([r1, r2], node.outputs)
        assert s == ["MyOp(R1, R2)"]

    def test_as_string_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        s = self.str([r1, r2, r5], node2.outputs)
        assert s == ["MyOp(MyOp(R1, R2), R5)"]

    def test_multiple_references(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(*1 -> MyOp(R1, R2), *1)"]

    def test_cutoff(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str(node.outputs, node2.outputs) == ["MyOp(R3, R3)"]
        assert self.str(node2.inputs, node2.outputs) == ["MyOp(R3, R3)"]


class TestClone(X):
    def test_accurate(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        _, new = clone([r1, r2], node.outputs, False)
        assert self.str([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        _, new = clone([r1, r2, r5], node2.outputs, False)
        assert (
            node2.outputs[0].type == new[0].type and node2.outputs[0] is not new[0]
        )  # the new output is like the old one but not the same object
        assert node2 is not new[0].owner  # the new output has a new owner
        assert new[0].owner.inputs[1] is r5  # the inputs are not copied
        assert (
            new[0].owner.inputs[0].type == node.outputs[0].type
            and new[0].owner.inputs[0] is not node.outputs[0]
        )  # check that we copied deeper too

    def test_not_destructive(self):
        # Checks that manipulating a cloned graph leaves the original unchanged.
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = MyVariable(7), MyVariable(8)
        assert self.str(inputs(new_node.outputs), new_node.outputs) == ["MyOp(R7, R8)"]
        assert self.str(inputs(node.outputs), node.outputs) == [
            "MyOp(MyOp(R1, R2), R5)"
        ]

    def test_constant(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = MyVariable(7), MyVariable(8)
        c1 = tensor.constant(1.5)

        i, o = clone([c1], [c1])
        assert i[0] is not c1 and o[0] is not c1

        i, o = clone([c1], [c1], False)
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], True, False)
        assert i[0] is not c1 and o[0] is not c1

        i, o = clone([c1], [c1], False, True)
        assert i[0] is c1 and o[0] is c1


def prenode(obj):
    if isinstance(obj, Variable):
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Apply):
        return obj.inputs


class TestToposort:
    def test_0(self):
        # Test a simple graph
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp.make_node(r1, r2)
        o2 = MyOp.make_node(o.outputs[0], r5)

        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r2, r1, o, o.outputs[0], o2, o2.outputs[0]]

        all = io_toposort([r5], o2.outputs)
        assert all == [o, o2]

    def test_1(self):
        # Test a graph with double dependencies
        r1, r5 = MyVariable(1), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        o2 = MyOp.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]]

    def test_2(self):
        # Test a graph where the inputs have owners
        r1, r5 = MyVariable(1), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        r2b = o.outputs[0]
        o2 = MyOp.make_node(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

        o2 = MyOp.make_node(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

    def test_3(self):
        # Test a graph which is not connected
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        assert all == [o1, o0] or all == [o0, o1]

    def test_4(self):
        # Test inputs and outputs mixed together in a chain graph
        r1, r2 = MyVariable(1), MyVariable(2)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        assert all == [o1]

    def test_5(self):
        # Test when outputs have clients
        r1, r2, r4 = MyVariable(1), MyVariable(2), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        MyOp.make_node(o0.outputs[0], r4)
        all = io_toposort([], o0.outputs)
        assert all == [o0]


class TestEval:
    def setup_method(self):
        self.x, self.y = tensor.scalars("x", "y")
        self.z = self.x + self.y
        self.w = 2 * self.z

    def test_eval(self):
        assert self.w.eval({self.x: 1.0, self.y: 2.0}) == 6.0
        assert self.w.eval({self.z: 3}) == 6.0
        assert hasattr(self.w, "_fn_cache"), "variable must have cache after eval"
        assert not hasattr(
            pickle.loads(pickle.dumps(self.w)), "_fn_cache"
        ), "temporary functions must not be serialized"


class TestAutoName:
    def test_auto_name(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1, r2 = MyVariable(1), MyVariable(2)
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)

    def test_constant(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = tensor.constant(1.5)
        assert r1.auto_name == "auto_" + str(autoname_id), (
            r1.auto_name,
            "auto_" + str(autoname_id),
        )

        r3 = tensor.constant(1.6)
        assert r3.auto_name == "auto_" + str(autoname_id + 1)

    def test_tensorvariable(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = tensor.TensorType(dtype="int32", broadcastable=())("myvar")
        r2 = tensor.TensorVariable(tensor.TensorType(dtype="int32", broadcastable=()))
        r3 = shared(np.random.randn(3, 4))
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)
        assert r3.auto_name == "auto_" + str(autoname_id + 2)

    @pytest.mark.skipif(
        not sparse.enable_sparse, reason="Optional package SciPy not installed"
    )
    def test_sparsevariable(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = sparse.csc_matrix(name="x", dtype="float32")
        r2 = sparse.dense_from_sparse(r1)
        r3 = sparse.csc_from_dense(r2)
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)
        assert r3.auto_name == "auto_" + str(autoname_id + 2)

    def test_randomvariable(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        mytype = tensor.TensorType(dtype="int32", broadcastable=())
        r1 = tensor.shared_randomstreams.RandomStateSharedVariable(
            name="x", type=mytype, value=1, strict=False
        )
        r2 = tensor.shared_randomstreams.RandomStateSharedVariable(
            name="x", type=mytype, value=1, strict=False
        )
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)

    def test_clone(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = MyVariable(1)
        r2 = r1.clone()
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)


def test_equal_computations():
    # This was a bug report by a Theano user.
    c = tensor.type_other.NoneConst
    assert equal_computations([c], [c])
    m = tensor.matrix()
    max_argmax1 = tensor.max_and_argmax(m)
    max_argmax2 = tensor.max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)

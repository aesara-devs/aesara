import pickle
from itertools import count

import numpy as np
import pytest

from aesara import config, function, shared
from aesara import tensor as at
from aesara.graph.basic import (
    Apply,
    Constant,
    NominalVariable,
    Variable,
    ancestors,
    applys_between,
    as_string,
    clone,
    clone_get_equiv,
    clone_replace,
    equal_computations,
    general_toposort,
    get_var_by_name,
    graph_inputs,
    io_toposort,
    is_in_ancestors,
    list_of_nodes,
    orphans_between,
    vars_between,
    walk,
)
from aesara.graph.op import Op
from aesara.graph.type import Type
from aesara.tensor.math import max_and_argmax
from aesara.tensor.type import (
    TensorType,
    dvector,
    fvector,
    iscalars,
    matrix,
    scalars,
    vector,
)
from aesara.tensor.type_other import NoneConst
from aesara.tensor.var import TensorVariable
from aesara.utils import HashableNDArray
from tests import unittest_tools as utt
from tests.graph.utils import MyInnerGraphOp


pytestmark = pytest.mark.filterwarnings("error")


class MyType(Type):
    def __init__(self, thingy):
        self.thingy = thingy

    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, MyType) and other.thingy == self.thingy

    def __hash__(self):
        return hash((type(self), self.thingy))

    def __str__(self):
        return f"R{self.thingy}"

    def __repr__(self):
        return f"R{self.thingy}"


def MyVariable(thingy):
    return Variable(MyType(thingy), None, None)


class MyOp(Op):

    __props__ = ()

    def make_node(self, *inputs):
        for input in inputs:
            assert isinstance(input, Variable)
            assert isinstance(input.type, MyType)
        outputs = [MyVariable(sum(input.type.thingy for input in inputs))]
        return Apply(self, list(inputs), outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")


my_op = MyOp()


def leaf_formatter(leaf):
    return str(leaf.type)


def node_formatter(node, argstrings):
    return f"{node.op}({', '.join(argstrings)})"


def format_graph(inputs, outputs):
    return as_string(
        inputs,
        outputs,
        leaf_formatter=leaf_formatter,
        node_formatter=node_formatter,
    )


class TestStr:
    def test_as_string(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = my_op.make_node(r1, r2)
        s = format_graph([r1, r2], node.outputs)
        assert s == ["MyOp(R1, R2)"]

    def test_as_string_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = my_op.make_node(r1, r2)
        node2 = my_op.make_node(node.outputs[0], r5)
        s = format_graph([r1, r2, r5], node2.outputs)
        assert s == ["MyOp(MyOp(R1, R2), R5)"]

    def test_multiple_references(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = my_op.make_node(r1, r2)
        node2 = my_op.make_node(node.outputs[0], node.outputs[0])
        assert format_graph([r1, r2, r5], node2.outputs) == [
            "MyOp(*1 -> MyOp(R1, R2), *1)"
        ]

    def test_cutoff(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = my_op.make_node(r1, r2)
        node2 = my_op.make_node(node.outputs[0], node.outputs[0])
        assert format_graph(node.outputs, node2.outputs) == ["MyOp(R3, R3)"]
        assert format_graph(node2.inputs, node2.outputs) == ["MyOp(R3, R3)"]


class TestClone:
    def test_accurate(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = my_op.make_node(r1, r2)
        _, new = clone([r1, r2], node.outputs, copy_inputs=False)
        assert format_graph([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = my_op.make_node(r1, r2)
        node2 = my_op.make_node(node.outputs[0], r5)
        _, new = clone([r1, r2, r5], node2.outputs, copy_inputs=False)
        assert node2.outputs[0].type == new[0].type and node2.outputs[0] is new[0]
        assert node2 is new[0].owner
        assert new[0].owner.inputs[1] is r5  # the inputs are not copied
        assert (
            new[0].owner.inputs[0].type == node.outputs[0].type
            and new[0].owner.inputs[0] is node.outputs[0]
        )

    def test_constant(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = my_op.make_node(my_op.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, copy_inputs=False)
        new_node = new[0].owner
        new_node.inputs = [MyVariable(7), MyVariable(8)]
        c1 = at.constant(1.5)

        i, o = clone([c1], [c1])
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], False)
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], True, False)
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], False, True)
        assert i[0] is c1 and o[0] is c1

    def test_clone_inner_graph(self):
        r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
        o1 = my_op(r1, r2)
        o1.name = "o1"

        # Inner graph
        igo_in_1 = MyVariable(4)
        igo_in_2 = MyVariable(5)
        igo_out_1 = my_op(igo_in_1, igo_in_2)
        igo_out_1.name = "igo1"

        igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

        o2 = igo(r3, o1)
        o2.name = "o1"

        o2_node = o2.owner
        o2_node_clone = o2_node.clone(clone_inner_graph=True)

        assert o2_node_clone is o2_node
        assert o2_node_clone.op.fgraph is o2_node.op.fgraph
        assert equal_computations(
            o2_node_clone.op.fgraph.outputs, o2_node.op.fgraph.outputs
        )


def prenode(obj):
    if isinstance(obj, Variable):
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Apply):
        return obj.inputs


class TestToposort:
    def test_simple(self):
        # Test a simple graph
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = my_op(r1, r2)
        o.name = "o1"
        o2 = my_op(o, r5)
        o2.name = "o2"

        clients = {}
        res = general_toposort([o2], prenode, clients=clients)

        assert clients == {
            o2.owner: [o2],
            o: [o2.owner],
            r5: [o2.owner],
            o.owner: [o],
            r1: [o.owner],
            r2: [o.owner],
        }
        assert res == [r5, r2, r1, o.owner, o, o2.owner, o2]

        with pytest.raises(ValueError):
            general_toposort(
                [o2], prenode, compute_deps_cache=lambda x: None, deps_cache=None
            )

        res = io_toposort([r5], [o2])
        assert res == [o.owner, o2.owner]

    def test_double_dependencies(self):
        # Test a graph with double dependencies
        r1, r5 = MyVariable(1), MyVariable(5)
        o = my_op.make_node(r1, r1)
        o2 = my_op.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]]

    def test_inputs_owners(self):
        # Test a graph where the inputs have owners
        r1, r5 = MyVariable(1), MyVariable(5)
        o = my_op.make_node(r1, r1)
        r2b = o.outputs[0]
        o2 = my_op.make_node(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

        o2 = my_op.make_node(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

    def test_not_connected(self):
        # Test a graph which is not connected
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = my_op.make_node(r1, r2)
        o1 = my_op.make_node(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        assert all == [o1, o0] or all == [o0, o1]

    def test_io_chain(self):
        # Test inputs and outputs mixed together in a chain graph
        r1, r2 = MyVariable(1), MyVariable(2)
        o0 = my_op.make_node(r1, r2)
        o1 = my_op.make_node(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        assert all == [o1]

    def test_outputs_clients(self):
        # Test when outputs have clients
        r1, r2, r4 = MyVariable(1), MyVariable(2), MyVariable(4)
        o0 = my_op.make_node(r1, r2)
        my_op.make_node(o0.outputs[0], r4)
        all = io_toposort([], o0.outputs)
        assert all == [o0]


@pytest.mark.skip(reason="Not finished")
class TestEval:
    def setup_method(self):
        self.x, self.y = scalars("x", "y")
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
        r1 = at.constant(1.5)
        assert r1.auto_name == "auto_" + str(autoname_id), (
            r1.auto_name,
            "auto_" + str(autoname_id),
        )

        r3 = at.constant(1.6)
        assert r3.auto_name == "auto_" + str(autoname_id + 1)

    def test_tensorvariable(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = TensorType(dtype="int32", shape=())("myvar")
        r2 = TensorVariable(TensorType(dtype="int32", shape=()), None)
        r3 = shared(np.random.standard_normal((3, 4)))
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)
        assert r3.auto_name == "auto_" + str(autoname_id + 2)

    def test_clone(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = MyVariable(1)
        r2 = r1.clone()
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)

        assert r1.name is None and r1.name is r2.name

        r3_name = "r3"
        r3 = r1.clone(name=r3_name)
        assert r3.name == r3_name


def test_equal_computations():

    a, b = iscalars(2)

    with pytest.raises(ValueError):
        equal_computations([a], [a, b])

    assert equal_computations([a], [a])
    assert equal_computations([at.as_tensor(1)], [at.as_tensor(1)])
    assert not equal_computations([b], [a])
    assert not equal_computations([at.as_tensor(1)], [at.as_tensor(2)])

    assert equal_computations([2], [2])
    assert equal_computations([np.r_[2, 1]], [np.r_[2, 1]])
    assert equal_computations([np.r_[2, 1]], [at.as_tensor(np.r_[2, 1])])
    assert equal_computations([at.as_tensor(np.r_[2, 1])], [np.r_[2, 1]])

    assert not equal_computations([2], [a])
    assert not equal_computations([np.r_[2, 1]], [a])
    assert not equal_computations([a], [2])
    assert not equal_computations([a], [np.r_[2, 1]])

    assert equal_computations([NoneConst], [NoneConst])

    m = matrix()
    max_argmax1 = max_and_argmax(m)
    max_argmax2 = max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)


def test_walk():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    def expand(r):
        if r.owner:
            return r.owner.inputs

    res = walk([o2], expand, bfs=True, return_children=False)
    res_list = list(res)
    assert res_list == [o2, r3, o1, r1, r2]

    res = walk([o2], expand, bfs=False, return_children=False)
    res_list = list(res)
    assert res_list == [o2, o1, r2, r1, r3]

    res = walk([o2], expand, bfs=True, return_children=True)
    res_list = list(res)
    assert res_list == [
        (o2, [r3, o1]),
        (r3, None),
        (o1, [r1, r2]),
        (r1, None),
        (r2, None),
    ]


def test_ancestors():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    res = ancestors([o2], blockers=None)
    res_list = list(res)
    assert res_list == [o2, r3, o1, r1, r2]

    res = ancestors([o2], blockers=None)
    assert r3 in res
    res_list = list(res)
    assert res_list == [o1, r1, r2]

    res = ancestors([o2], blockers=[o1])
    res_list = list(res)
    assert res_list == [o2, r3, o1]


def test_graph_inputs():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    res = graph_inputs([o2], blockers=None)
    res_list = list(res)
    assert res_list == [r3, r1, r2]


def test_variables_and_orphans():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    vars_res = vars_between([r1, r2], [o2])
    orphans_res = orphans_between([r1, r2], [o2])

    vars_res_list = list(vars_res)
    orphans_res_list = list(orphans_res)
    assert vars_res_list == [o2, o1, r3, r2, r1]
    assert orphans_res_list == [r3]


def test_ops():

    r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, r4)
    o2.name = "o2"
    o3 = my_op(r3, o1, o2)
    o3.name = "o3"

    res = applys_between([r1, r2], [o3])
    res_list = list(res)
    assert res_list == [o3.owner, o2.owner, o1.owner]


def test_list_of_nodes():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    res = list_of_nodes([r1, r2], [o2])
    assert res == [o2.owner, o1.owner]


def test_is_in_ancestors():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"
    o2 = my_op(r3, o1)
    o2.name = "o2"

    assert is_in_ancestors(o2.owner, o1.owner)


@pytest.mark.xfail(reason="Not implemented")
def test_io_connection_pattern():
    raise AssertionError()


@pytest.mark.xfail(reason="Not implemented")
def test_view_roots():
    raise AssertionError()


def test_get_var_by_name():

    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = my_op(r1, r2)
    o1.name = "o1"

    # Inner graph
    igo_in_1 = MyVariable(4)
    igo_in_2 = MyVariable(5)
    igo_out_1 = my_op(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    o2 = igo(r3, o1)
    o2.name = "o1"

    res = get_var_by_name([o1, o2], "blah")

    assert res == ()

    res = get_var_by_name([o1, o2], "o1")

    assert set(res) == {o1, o2}

    (res,) = get_var_by_name([o1, o2], o1.auto_name)

    assert res == o1

    (res,) = get_var_by_name([o1, o2], "igo1")

    exp_res = igo.fgraph.outputs[0]
    assert res == exp_res


class TestCloneReplace:
    def test_cloning_no_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=None, rebuild_strict=True, copy_inputs_over=True)
        f2_inp = graph_inputs([f2])

        assert z in f2_inp
        assert x in f2_inp
        assert y in f2_inp

    def test_cloning_no_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=None, rebuild_strict=True, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])

        assert z not in f2_inp
        assert x not in f2_inp
        assert y not in f2_inp

    def test_cloning_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace={y: y2}, rebuild_strict=True, copy_inputs_over=True
        )
        f2_inp = graph_inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_not_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace={y: y2}, rebuild_strict=False, copy_inputs_over=True
        )
        f2_inp = graph_inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=[(y, y2)], rebuild_strict=True, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    def test_cloning_replace_not_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=[(y, y2)], rebuild_strict=False, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    @pytest.mark.skip(reason="Not finished")
    def test_clone(self):
        def test(x, y, mention_y):
            if mention_y:
                d = 0.1 + 0 * y
            else:
                d = 0.1
            out = clone_replace(y, replace={x: x + d})
            return function([], out)()

        x = shared(np.asarray(0.0, dtype=config.floatX))
        utt.assert_allclose(
            test(x, at.sum((x + 1) ** 2), mention_y=False), 1.21000003815
        )
        utt.assert_allclose(
            test(x, at.sum((x + 1) ** 2), mention_y=True), 1.21000003815
        )


def test_clone_new_inputs():
    """Make sure that `Apply.clone_with_new_inputs` properly handles `Type` changes."""

    x = at.tensor(np.float64, shape=(None,))
    y = at.tensor(np.float64, shape=(1,))

    z = at.add(x, y)
    assert z.type.shape == (None,)

    x_new = at.tensor(np.float64, shape=(1,))

    # The output nodes should be reconstructed, because the input types' static
    # shape information increased in specificity
    z_node_new = z.owner.clone_with_new_inputs([x_new, y])

    assert z_node_new.outputs[0].type.shape == (1,)
    assert z_node_new.inputs[0].type.shape == (1,)
    assert z_node_new.inputs[1].type.shape == (1,)

    # Now, attempt to decrease the specificity of the first input's static
    # shape information, but, because we're using strict conversion, we
    # shouldn't lose any information
    z = at.add(x_new, y)
    assert z.type.shape == (1,)

    z_node_new = z.owner.clone_with_new_inputs([x, y], strict=True)

    assert z_node_new.outputs[0].type.shape == (1,)
    assert z_node_new.inputs[0].type.shape == (1,)
    assert z_node_new.inputs[1].type.shape == (1,)


def test_clone_get_equiv():
    x = vector("x")
    y = vector("y")
    z = vector("z")
    a = x * y
    a_node = a.owner
    b = a + 1.0

    memo = {a: z}
    _ = clone_get_equiv([x, y], [b], copy_inputs=False, copy_orphans=False, memo=memo)

    assert x in memo
    assert y in memo
    assert memo[a] is z
    # All the outputs of `a` already had replacements/clones in the map, so
    # there is no need to re-clone it (unless another replacement/clone
    # re-introduces `a.owner` somehow).
    assert a_node not in memo
    assert equal_computations([memo[b]], [z + 1.0])


def test_NominalVariable():

    type1 = MyType(1)

    nv1 = NominalVariable(1, type1)
    nv2 = NominalVariable(1, type1)

    assert nv1 is nv2
    assert nv1.equals(nv2)
    assert hash(nv1) == hash(nv2)

    type2 = MyType(2)
    nv3 = NominalVariable(1, type2)

    assert not nv1.equals(nv3)
    assert hash(nv1) != hash(nv3)

    type3 = MyType(1)

    assert type3 == type1

    nv4 = NominalVariable(1, type3)

    assert nv1 is nv4
    assert nv1.equals(nv4)
    assert hash(nv1) == hash(nv4)

    nv5 = NominalVariable(2, type3)
    assert not nv4.equals(nv5)
    assert hash(nv4) != hash(nv5)

    assert repr(nv5) == f"NominalVariable(2, {repr(type3)})"

    assert hash(nv5) == hash((type(nv5), 2, type3))

    nv5_pkld = pickle.dumps(nv5)
    nv5_unpkld = pickle.loads(nv5_pkld)

    assert type(nv5_unpkld) is type(nv5)
    assert nv5_unpkld.equals(nv5)
    assert nv5_unpkld is nv5

    nv5_clone = nv5.clone()
    assert type(nv5_clone) is type(nv5)
    assert nv5_clone.equals(nv5)
    assert nv5_clone is nv5


def test_NominalVariable_create_variable_type():

    ttype = TensorType("float64", (None, None))
    ntv = NominalVariable(0, ttype)

    assert isinstance(ntv, TensorVariable)
    assert isinstance(ntv, NominalVariable)
    assert ntv.ndim == 2
    assert ntv.broadcastable == (False, False)
    assert ntv.dtype == "float64"

    ntv2 = NominalVariable(0, ttype)

    assert type(ntv2) is type(ntv)
    assert ntv2.equals(ntv)
    assert ntv2 is ntv

    ntv_pkld = pickle.dumps(ntv)
    ntv_unpkld = pickle.loads(ntv_pkld)

    assert type(ntv_unpkld) is type(ntv)
    assert ntv_unpkld == ntv
    assert ntv_unpkld is ntv


def test_Apply_equivalence():

    type1 = MyType(1)

    in_1 = Variable(type1, None, name="in_1")
    in_2 = Variable(type1, None, name="in_2")
    out_10 = Variable(type1, None, name="out_10")
    out_11 = Variable(type1, None, name="out_11")
    out_12 = Variable(type1, None, name="out_12")

    apply_1 = Apply(my_op, [in_1], [out_10])
    apply_2 = Apply(my_op, [in_1], [out_11])
    apply_3 = Apply(my_op, [in_2], [out_12])

    assert apply_1 is apply_2
    assert apply_1 == apply_2
    assert apply_1 != apply_3
    assert hash(apply_1) == hash(apply_2)
    assert hash(apply_1) != hash(apply_3)

    assert apply_1.inputs == apply_2.inputs

    assert apply_1.outputs == [out_10]
    assert apply_2.outputs == [out_10]
    # Output `Variable`s should be updated when the constructor is called with
    # the same inputs but different outputs.
    assert out_10.owner is apply_1
    assert out_11.owner is apply_1

    apply_1_pkl = pickle.dumps(apply_1)
    apply_1_2 = pickle.loads(apply_1_pkl)

    assert apply_1.op == apply_1_2.op
    assert len(apply_1.inputs) == len(apply_1_2.inputs)
    assert len(apply_1.outputs) == len(apply_1_2.outputs)
    assert apply_1.inputs[0].type == apply_1_2.inputs[0].type
    assert apply_1.inputs[0].name == apply_1_2.inputs[0].name
    assert apply_1.outputs[0].type == apply_1_2.outputs[0].type
    assert apply_1.outputs[0].name == apply_1_2.outputs[0].name


class MyType2(MyType):
    def filter(self, value, **kwargs):
        value = np.asarray(value).view(HashableNDArray)
        value.setflags(write=0)
        return value


def test_Constant_equivalence():
    type1 = MyType2(1)
    x = Constant(type1, 1.0)
    y = Constant(type1, 1.0)

    assert x == y
    assert x is y

    rng = np.random.default_rng(3209)
    a_val = rng.normal(size=(2, 3))
    c_val = rng.normal(size=(2, 3))

    a = Constant(type1, a_val)
    b = Constant(type1, a_val)
    c = Constant(type1, c_val)

    assert a == b
    assert a is b
    assert a != x
    assert a != c

    a_pkl = pickle.dumps(a)
    a_2 = pickle.loads(a_pkl)

    assert a.type == a_2.type
    assert a.data == a_2.data

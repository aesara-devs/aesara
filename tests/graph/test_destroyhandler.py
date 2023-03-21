import pickle
from copy import copy

import pytest

from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, Variable, clone
from aesara.graph.destroyhandler import DestroyHandler, fast_inplace_check
from aesara.graph.features import ReplaceValidate
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.rewriting.basic import (
    NodeProcessingGraphRewriter,
    OpKeyGraphRewriter,
    PatternNodeRewriter,
    SubstitutionNodeRewriter,
    WalkingGraphRewriter,
)
from aesara.graph.type import Type
from aesara.graph.utils import InconsistencyError
from tests.unittest_tools import assertFailure_fast


def OpKeyPatternNodeRewriter(p1, p2, ign=True):
    return OpKeyGraphRewriter(PatternNodeRewriter(p1, p2), ignore_newtrees=ign)


def TopoSubstitutionNodeRewriter(
    op1, op2, fail=NodeProcessingGraphRewriter.warn_ignore, ign=True
):
    return WalkingGraphRewriter(
        SubstitutionNodeRewriter(op1, op2), ignore_newtrees=ign, failure_callback=fail
    )


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)


def MyVariable(name):
    return Variable(MyType(), None, None, name=name)


def MyConstant(data):
    return Constant(MyType(), data=data)


class MyOp(Op):
    def __init__(
        self,
        nin,
        name,
        vmap=None,
        dmap=None,
        nout=1,
        destroyhandler_tolerate_same=None,
        destroyhandler_tolerate_aliased=None,
    ):
        if vmap is None:
            vmap = {}
        if dmap is None:
            dmap = {}
        if destroyhandler_tolerate_same is None:
            destroyhandler_tolerate_same = []
        if destroyhandler_tolerate_aliased is None:
            destroyhandler_tolerate_aliased = []

        self.nin = nin
        self.nout = nout
        self.name = name
        self.destroy_map = dmap
        self.view_map = vmap
        self.destroyhandler_tolerate_same = destroyhandler_tolerate_same
        self.destroyhandler_tolerate_aliased = destroyhandler_tolerate_aliased

    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyVariable(self.name + "_R") for i in range(self.nout)]
        return Apply(self, inputs, outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")

    def __str__(self):
        return self.name


sigmoid = MyOp(1, "Sigmoid")
transpose_view = MyOp(1, "TransposeView", vmap={0: [0]})
add = MyOp(2, "Add")
add_in_place = MyOp(2, "AddInPlace", dmap={0: [0]})
add_in_place_2 = MyOp(
    2, "AddInPlace", dmap={0: [0]}, destroyhandler_tolerate_same=[(0, 1)]
)
add_in_place_3 = MyOp(
    2, "AddInPlace", dmap={0: [0]}, destroyhandler_tolerate_aliased=[(0, 1)]
)
dot = MyOp(2, "Dot")
multiple = MyOp(2, "Multiple", nout=2)
multiple_in_place_0 = MyOp(2, "MultipleInPlace0", nout=2, dmap={0: [0]})
multiple_in_place_1 = MyOp(2, "MultipleInPlace1", nout=2, dmap={1: [1]})
multiple_in_place_0_1 = MyOp(2, "MultipleInPlace01", nout=2, dmap={0: [0], 1: [1]})


def inputs():
    x = MyVariable("x")
    y = MyVariable("y")
    z = MyVariable("z")
    return x, y, z


def create_fgraph(inputs, outputs, validate=True, algo=None):
    e = FunctionGraph(inputs, outputs, clone=False)
    e.attach_feature(DestroyHandler(algo=algo))
    e.attach_feature(ReplaceValidate())
    if validate:
        e.validate()
    return e


class FailureWatch:
    def __init__(self):
        self.failures = 0

    def __call__(self, exc, nav, pairs, lrewrite, node):
        assert isinstance(exc, InconsistencyError)
        self.failures += 1


def test_misc():
    x, y, z = inputs()
    e = transpose_view(transpose_view(transpose_view(transpose_view(x))))
    g = create_fgraph([x, y, z], [e])
    assert g.consistent()

    OpKeyPatternNodeRewriter((transpose_view, (transpose_view, "x")), "x").rewrite(g)

    assert str(g) == "FunctionGraph(x)"

    new_e = add(x, y)
    g.replace_validate(x, new_e)
    assert str(g) == "FunctionGraph(Add(x, y))"

    g.replace(new_e, dot(add_in_place(x, y), transpose_view(x)))
    assert str(g) == "FunctionGraph(Dot(AddInPlace(x, y), TransposeView(x)))"
    assert not g.consistent()

    (dh,) = (f for f in g._features if isinstance(f, DestroyHandler))
    g.remove_feature(dh)
    assert not hasattr(g, "destroyers")


@assertFailure_fast
def test_aliased_inputs_replacement():
    x, y, z = inputs()
    tv = transpose_view(x)
    tvv = transpose_view(tv)
    sx = sigmoid(x)
    e = add_in_place(x, tv)
    g = create_fgraph([x, y], [e], False)
    assert not g.consistent()
    g.replace(tv, sx)
    assert g.consistent()
    g.replace(sx, tv)
    assert not g.consistent()
    g.replace(tv, tvv)
    assert not g.consistent()
    g.replace(tv, sx)
    assert g.consistent()


@pytest.mark.parametrize("algo", [None, "fast"])
def test_indestructible(algo):
    x, y, z = inputs()
    x.tag.indestructible = True
    x = copy(x)
    # checking if indestructible survives the copy!
    assert x.tag.indestructible
    e = add_in_place(x, y)
    g = create_fgraph([x, y, z], [e], False, algo=algo)
    assert not g.consistent()
    g.replace_validate(e, add(x, y))
    assert g.consistent()


@assertFailure_fast
def test_usage_loop_through_views_2():
    x, y, z = inputs()
    e0 = transpose_view(transpose_view(sigmoid(x)))
    e = dot(add_in_place(x, y), transpose_view(e0))
    g = create_fgraph([x, y, z], [e])
    assert g.consistent()  # because sigmoid can do the copy
    g.replace(e0, x)
    assert not g.consistent()  # we cut off the path to the sigmoid


@assertFailure_fast
def test_destroyers_loop():
    # AddInPlace(x, y) and AddInPlace(y, x) should not coexist
    x, y, z = inputs()
    e1 = add(x, y)
    e2 = add(y, x)
    g = create_fgraph([x, y, z], [e1, e2])
    assert g.consistent()
    g.replace_validate(e1, add_in_place(x, y))
    assert g.consistent()
    with pytest.raises(InconsistencyError):
        g.replace_validate(e2, add_in_place(y, x))
    assert g.consistent()

    x, y, z = inputs()
    e1 = add(x, y)
    e2 = add(y, x)
    g = create_fgraph([x, y, z], [e1, e2])
    assert g.consistent()
    g.replace_validate(e2, add_in_place(y, x))
    assert g.consistent()
    with pytest.raises(InconsistencyError):
        g.replace_validate(e1, add_in_place(x, y))
    assert g.consistent()


def test_aliased_inputs():
    x, y, z = inputs()
    e = add_in_place(x, x)
    g = create_fgraph([x], [e], False)
    assert not g.consistent()


def test_aliased_inputs2():
    x, y, z = inputs()
    e = add_in_place(x, transpose_view(x))
    g = create_fgraph([x], [e], False)
    assert not g.consistent()


@assertFailure_fast
def test_aliased_inputs_tolerate():
    x, y, z = inputs()
    e = add_in_place_2(x, x)
    g = create_fgraph([x], [e], False)
    assert g.consistent()


def test_aliased_inputs_tolerate2():
    x, y, z = inputs()
    e = add_in_place_2(x, transpose_view(x))
    g = create_fgraph([x], [e], False)
    assert not g.consistent()


@assertFailure_fast
def test_same_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, x)
    g = create_fgraph([x], [e], False)
    assert g.consistent()


@assertFailure_fast
def test_different_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, transpose_view(x))
    g = create_fgraph([x], [e], False)
    assert g.consistent()
    # warning - don't run this because it would produce the wrong answer
    # add_in_place_3 is actually not correct when aliasing of inputs
    # is ignored.


def test_indestructible_through_views():
    x, y, z = inputs()
    x.tag.indestructible = True
    tv = transpose_view(x)
    e = add_in_place(tv, y)
    g = create_fgraph([x, y, z], [e], False)
    assert not g.consistent()
    g.replace_validate(tv, sigmoid(x))
    assert g.consistent()


def test_indirect():
    x, y, z = inputs()
    e0 = add_in_place(x, y)
    e = dot(sigmoid(e0), transpose_view(x))
    g = create_fgraph([x, y, z], [e], False)
    assert not g.consistent()
    new_e0 = add(x, y)
    g.replace(e0, new_e0)
    assert g.consistent()
    g.replace(new_e0, add_in_place(x, y))
    assert not g.consistent()


@assertFailure_fast
def test_indirect_2():
    x, y, z = inputs()
    e0 = transpose_view(x)
    e = dot(sigmoid(add_in_place(x, y)), e0)
    g = create_fgraph([x, y, z], [e], False)
    assert not g.consistent()
    new_e0 = add(e0, y)
    g.replace(e0, new_e0)
    assert g.consistent()


@assertFailure_fast
def test_long_destroyers_loop():
    x, y, z = inputs()
    e = dot(dot(add_in_place(x, y), add_in_place(y, z)), add(z, x))
    g = create_fgraph([x, y, z], [e])
    assert g.consistent()
    TopoSubstitutionNodeRewriter(add, add_in_place).rewrite(g)
    assert g.consistent()
    # we don't want to see that!
    assert (
        str(g)
        != "FunctionGraph(Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x)))"
    )
    e2 = dot(dot(add_in_place(x, y), add_in_place(y, z)), add_in_place(z, x))
    with pytest.raises(InconsistencyError):
        create_fgraph(*clone([x, y, z], [e2]))


def test_misc_2():
    x, y, z = inputs()
    tv = transpose_view(x)
    e = add_in_place(x, tv)
    g = create_fgraph([x, y], [e], False)
    assert not g.consistent()
    g.replace(tv, x)
    assert not g.consistent()


def test_multi_destroyers():
    x, y, z = inputs()
    e = add(add_in_place(x, y), add_in_place(x, y))
    with pytest.raises(InconsistencyError):
        create_fgraph([x, y, z], [e])


@assertFailure_fast
def test_multi_destroyers_through_views():
    x, y, z = inputs()
    e = dot(add(transpose_view(z), y), add(z, x))
    g = create_fgraph([x, y, z], [e])
    assert g.consistent()
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(add, add_in_place, fail).rewrite(g)
    assert g.consistent()
    assert fail.failures == 1  # should have succeeded once and failed once


def test_repair_destroy_path():
    x, y, z = inputs()
    e1 = transpose_view(transpose_view(x))
    e2 = transpose_view(transpose_view(e1))
    e3 = add_in_place(e2, y)
    e4 = add_in_place(e1, z)
    g = create_fgraph([x, y, z], [e3, e4], False)
    assert not g.consistent()
    g.replace(e2, transpose_view(x))
    assert not g.consistent()


def test_usage_loop():
    x, y, z = inputs()
    g = create_fgraph([x, y, z], [dot(add_in_place(x, z), x)], False)
    assert not g.consistent()
    # replace add_in_place with add
    TopoSubstitutionNodeRewriter(add_in_place, add).rewrite(g)
    assert g.consistent()


def test_usage_loop_through_views():
    x, y, z = inputs()
    aip = add_in_place(x, y)
    e = dot(aip, transpose_view(x))
    g = create_fgraph([x, y, z], [e], False)
    assert not g.consistent()
    g.replace_validate(aip, add(x, z))
    assert g.consistent()


@assertFailure_fast
def test_usage_loop_insert_views():
    x, y, z = inputs()
    e = dot(add_in_place(x, add(y, z)), sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x))))))
    g = create_fgraph([x, y, z], [e])
    assert g.consistent()
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(sigmoid, transpose_view, fail).rewrite(g)
    assert g.consistent()
    # it must keep one sigmoid in the long sigmoid chain
    assert fail.failures == 1


def test_value_repl():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = create_fgraph([x, y], [e], False)
    assert g.consistent()
    g.replace(sy, MyConstant("abc"))
    assert g.consistent()


@config.change_flags(compute_test_value="off")
def test_value_repl_2():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = create_fgraph([x, y], [e], False)
    assert g.consistent()
    g.replace(sy, transpose_view(MyConstant("abc")))
    assert g.consistent()


@assertFailure_fast
def test_multiple_inplace():
    # this tests issue #5223
    # there were some problems with Ops that have more than
    # one in-place input.
    x, y, z = inputs()
    # we will try to replace this op with an in-place version
    m = multiple(x, y)
    # this makes it impossible to run in-place on x
    e_1 = dot(m[0], x)
    # try to confuse the DestroyHandler: this dot Op can run
    # before multiple and then multiple can still run in-place on y
    e_2 = dot(y, y)
    g = create_fgraph([x, y], [e_1, e_2], False)
    assert g.consistent()

    # try to work in-place on x/0 and y/1 (this should fail)
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(multiple, multiple_in_place_0_1, fail).rewrite(g)
    assert g.consistent()
    assert fail.failures == 1

    # try to work in-place on x/0 (this should fail)
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(multiple, multiple_in_place_0, fail).rewrite(g)
    assert g.consistent()
    assert fail.failures == 1

    # try to work in-place on y/1 (this should succeed)
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(multiple, multiple_in_place_1, fail).rewrite(g)
    assert g.consistent()
    assert fail.failures == 0

    # try to work in-place on x/0 and y/1 (this should still fail)
    fail = FailureWatch()
    TopoSubstitutionNodeRewriter(
        multiple_in_place_1, multiple_in_place_0_1, fail
    ).rewrite(g)
    assert g.consistent()
    assert fail.failures == 1


def test_pickle():
    x, y, z = inputs()
    tv = transpose_view(x)
    e = add_in_place(x, tv)
    fg = create_fgraph([x, y], [e], False)
    assert not fg.consistent()

    fg_pkld = pickle.dumps(fg)
    fg_unpkld = pickle.loads(fg_pkld)

    assert any(isinstance(ft, DestroyHandler) for ft in fg_unpkld._features)
    assert all(hasattr(fg, attr) for attr in ("_destroyhandler_destroyers",))


def test_fast_inplace_check():
    x, y = MyVariable("x"), MyVariable("y")
    e = add_in_place(x, y)
    fg = FunctionGraph(outputs=[e], clone=False)
    fg.attach_feature(DestroyHandler())

    res = fast_inplace_check(fg, fg.inputs)
    assert res == [y]


def test_fast_destroy():
    """Make sure `DestroyHandler.fast_destroy` catches basic inconsistencies."""
    x, y, z = MyVariable("x"), MyVariable("y"), MyVariable("z")

    w = add_in_place(x, dot(y, x))
    with pytest.raises(InconsistencyError):
        create_fgraph([x, y], [w], algo="fast")

    w = add_in_place(x, y)
    w = add_in_place(w, z)
    with pytest.raises(InconsistencyError):
        create_fgraph([x, y, z], [w], algo="fast")

    w = transpose_view(x)
    w = add_in_place(w, y)
    with pytest.raises(InconsistencyError):
        create_fgraph([x, y], [w], algo="fast")

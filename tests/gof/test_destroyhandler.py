from copy import copy

import pytest

from tests.unittest_tools import assertFailure_fast
from theano import change_flags
from theano.gof import destroyhandler, graph
from theano.gof.fg import FunctionGraph, InconsistencyError
from theano.gof.graph import Apply, Variable
from theano.gof.op import Op
from theano.gof.opt import (
    NavigatorOptimizer,
    OpKeyOptimizer,
    OpSub,
    PatternSub,
    TopoOptimizer,
)
from theano.gof.toolbox import ReplaceValidate
from theano.gof.type import Type


def PatternOptimizer(p1, p2, ign=True):
    return OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)


def OpSubOptimizer(op1, op2, fail=NavigatorOptimizer.warn_ignore, ign=True):
    return TopoOptimizer(OpSub(op1, op2), ignore_newtrees=ign, failure_callback=fail)


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
    return graph.Constant(MyType(), data=data)


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


def Env(inputs, outputs, validate=True):
    e = FunctionGraph(inputs, outputs, clone=False)
    e.attach_feature(destroyhandler.DestroyHandler())
    e.attach_feature(ReplaceValidate())
    if validate:
        e.validate()
    return e


class FailureWatch:
    # when passed to OpSubOptimizer or PatternOptimizer, counts the
    # number of failures
    def __init__(self):
        self.failures = 0

    def __call__(self, exc, nav, pairs, lopt, node):
        assert isinstance(exc, InconsistencyError)
        self.failures += 1


#################
# Test protocol #
#################


def test_misc():
    x, y, z = inputs()
    e = transpose_view(transpose_view(transpose_view(transpose_view(x))))
    g = Env([x, y, z], [e])
    assert g.consistent()
    PatternOptimizer((transpose_view, (transpose_view, "x")), "x").optimize(g)
    assert str(g) == "[x]"
    new_e = add(x, y)
    g.replace_validate(x, new_e)
    assert str(g) == "[Add(x, y)]"
    g.replace(new_e, dot(add_in_place(x, y), transpose_view(x)))
    assert str(g) == "[Dot(AddInPlace(x, y), TransposeView(x))]"
    assert not g.consistent()


######################
# Test protocol skip #
######################


@assertFailure_fast
def test_aliased_inputs_replacement():
    x, y, z = inputs()
    tv = transpose_view(x)
    tvv = transpose_view(tv)
    sx = sigmoid(x)
    e = add_in_place(x, tv)
    g = Env([x, y], [e], False)
    assert not g.consistent()
    g.replace(tv, sx)
    assert g.consistent()
    g.replace(sx, tv)
    assert not g.consistent()
    g.replace(tv, tvv)
    assert not g.consistent()
    g.replace(tv, sx)
    assert g.consistent()


def test_indestructible():
    x, y, z = inputs()
    x.tag.indestructible = True
    x = copy(x)
    # checking if indestructible survives the copy!
    assert x.tag.indestructible
    e = add_in_place(x, y)
    g = Env([x, y, z], [e], False)
    assert not g.consistent()
    g.replace_validate(e, add(x, y))
    assert g.consistent()


@assertFailure_fast
def test_usage_loop_through_views_2():
    x, y, z = inputs()
    e0 = transpose_view(transpose_view(sigmoid(x)))
    e = dot(add_in_place(x, y), transpose_view(e0))
    g = Env([x, y, z], [e])
    assert g.consistent()  # because sigmoid can do the copy
    g.replace(e0, x)
    assert not g.consistent()  # we cut off the path to the sigmoid


@assertFailure_fast
def test_destroyers_loop():
    # AddInPlace(x, y) and AddInPlace(y, x) should not coexist
    x, y, z = inputs()
    e1 = add(x, y)
    e2 = add(y, x)
    g = Env([x, y, z], [e1, e2])
    assert g.consistent()
    g.replace_validate(e1, add_in_place(x, y))
    assert g.consistent()
    with pytest.raises(InconsistencyError):
        g.replace_validate(e2, add_in_place(y, x))
    assert g.consistent()

    x, y, z = inputs()
    e1 = add(x, y)
    e2 = add(y, x)
    g = Env([x, y, z], [e1, e2])
    assert g.consistent()
    g.replace_validate(e2, add_in_place(y, x))
    assert g.consistent()
    with pytest.raises(InconsistencyError):
        g.replace_validate(e1, add_in_place(x, y))
    assert g.consistent()


########
# Misc #
########


def test_aliased_inputs():
    x, y, z = inputs()
    e = add_in_place(x, x)
    g = Env([x], [e], False)
    assert not g.consistent()


def test_aliased_inputs2():
    x, y, z = inputs()
    e = add_in_place(x, transpose_view(x))
    g = Env([x], [e], False)
    assert not g.consistent()


@assertFailure_fast
def test_aliased_inputs_tolerate():
    x, y, z = inputs()
    e = add_in_place_2(x, x)
    g = Env([x], [e], False)
    assert g.consistent()


def test_aliased_inputs_tolerate2():
    x, y, z = inputs()
    e = add_in_place_2(x, transpose_view(x))
    g = Env([x], [e], False)
    assert not g.consistent()


@assertFailure_fast
def test_same_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, x)
    g = Env([x], [e], False)
    assert g.consistent()


@assertFailure_fast
def test_different_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, transpose_view(x))
    g = Env([x], [e], False)
    assert g.consistent()
    # warning - don't run this because it would produce the wrong answer
    # add_in_place_3 is actually not correct when aliasing of inputs
    # is ignored.


def test_indestructible_through_views():
    x, y, z = inputs()
    x.tag.indestructible = True
    tv = transpose_view(x)
    e = add_in_place(tv, y)
    g = Env([x, y, z], [e], False)
    assert not g.consistent()
    g.replace_validate(tv, sigmoid(x))
    assert g.consistent()


def test_indirect():
    x, y, z = inputs()
    e0 = add_in_place(x, y)
    e = dot(sigmoid(e0), transpose_view(x))
    g = Env([x, y, z], [e], False)
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
    g = Env([x, y, z], [e], False)
    assert not g.consistent()
    new_e0 = add(e0, y)
    g.replace(e0, new_e0)
    assert g.consistent()


@assertFailure_fast
def test_long_destroyers_loop():
    x, y, z = inputs()
    e = dot(dot(add_in_place(x, y), add_in_place(y, z)), add(z, x))
    g = Env([x, y, z], [e])
    assert g.consistent()
    OpSubOptimizer(add, add_in_place).optimize(g)
    assert g.consistent()
    # we don't want to see that!
    assert str(g) != "[Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x))]"
    e2 = dot(dot(add_in_place(x, y), add_in_place(y, z)), add_in_place(z, x))
    with pytest.raises(InconsistencyError):
        Env(*graph.clone([x, y, z], [e2]))


def test_misc_2():
    x, y, z = inputs()
    tv = transpose_view(x)
    e = add_in_place(x, tv)
    g = Env([x, y], [e], False)
    assert not g.consistent()
    g.replace(tv, x)
    assert not g.consistent()


def test_multi_destroyers():
    x, y, z = inputs()
    e = add(add_in_place(x, y), add_in_place(x, y))
    with pytest.raises(InconsistencyError):
        Env([x, y, z], [e])


@assertFailure_fast
def test_multi_destroyers_through_views():
    x, y, z = inputs()
    e = dot(add(transpose_view(z), y), add(z, x))
    g = Env([x, y, z], [e])
    assert g.consistent()
    fail = FailureWatch()
    OpSubOptimizer(add, add_in_place, fail).optimize(g)
    assert g.consistent()
    assert fail.failures == 1  # should have succeeded once and failed once


def test_repair_destroy_path():
    x, y, z = inputs()
    e1 = transpose_view(transpose_view(x))
    e2 = transpose_view(transpose_view(e1))
    e3 = add_in_place(e2, y)
    e4 = add_in_place(e1, z)
    g = Env([x, y, z], [e3, e4], False)
    assert not g.consistent()
    g.replace(e2, transpose_view(x))
    assert not g.consistent()


def test_usage_loop():
    x, y, z = inputs()
    g = Env([x, y, z], [dot(add_in_place(x, z), x)], False)
    assert not g.consistent()
    # replace add_in_place with add
    OpSubOptimizer(add_in_place, add).optimize(g)
    assert g.consistent()


def test_usage_loop_through_views():
    x, y, z = inputs()
    aip = add_in_place(x, y)
    e = dot(aip, transpose_view(x))
    g = Env([x, y, z], [e], False)
    assert not g.consistent()
    g.replace_validate(aip, add(x, z))
    assert g.consistent()


@assertFailure_fast
def test_usage_loop_insert_views():
    x, y, z = inputs()
    e = dot(add_in_place(x, add(y, z)), sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x))))))
    g = Env([x, y, z], [e])
    assert g.consistent()
    fail = FailureWatch()
    OpSubOptimizer(sigmoid, transpose_view, fail).optimize(g)
    assert g.consistent()
    # it must keep one sigmoid in the long sigmoid chain
    assert fail.failures == 1


def test_value_repl():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = Env([x, y], [e], False)
    assert g.consistent()
    g.replace(sy, MyConstant("abc"))
    assert g.consistent()


@change_flags(compute_test_value="off")
def test_value_repl_2():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = Env([x, y], [e], False)
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
    g = Env([x, y], [e_1, e_2], False)
    assert g.consistent()

    # try to work in-place on x/0 and y/1 (this should fail)
    fail = FailureWatch()
    OpSubOptimizer(multiple, multiple_in_place_0_1, fail).optimize(g)
    assert g.consistent()
    assert fail.failures == 1

    # try to work in-place on x/0 (this should fail)
    fail = FailureWatch()
    OpSubOptimizer(multiple, multiple_in_place_0, fail).optimize(g)
    assert g.consistent()
    assert fail.failures == 1

    # try to work in-place on y/1 (this should succeed)
    fail = FailureWatch()
    OpSubOptimizer(multiple, multiple_in_place_1, fail).optimize(g)
    assert g.consistent()
    assert fail.failures == 0

    # try to work in-place on x/0 and y/1 (this should still fail)
    fail = FailureWatch()
    OpSubOptimizer(multiple_in_place_1, multiple_in_place_0_1, fail).optimize(g)
    assert g.consistent()
    assert fail.failures == 1

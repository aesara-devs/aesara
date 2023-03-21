import copy
import pickle

import numpy as np
import pytest

import aesara.tensor as at
from aesara.compile import shared
from aesara.compile.debugmode import DebugMode, InvalidValueError
from aesara.compile.function import function
from aesara.compile.function.types import Supervisor, UnusedInputError
from aesara.compile.io import In, Out
from aesara.compile.mode import Mode, get_default_mode
from aesara.compile.ops import update_placeholder
from aesara.configdefaults import config
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import OpKeyGraphRewriter, PatternNodeRewriter
from aesara.graph.utils import MissingInputError
from aesara.link.vm import VMLinker
from aesara.tensor.math import dot
from aesara.tensor.math import sum as at_sum
from aesara.tensor.math import tanh
from aesara.tensor.type import (
    dmatrix,
    dscalar,
    dscalars,
    dvector,
    fscalar,
    iscalar,
    matrix,
    scalar,
    scalars,
    vector,
)
from aesara.utils import exc_message
from tests.graph.utils import MyVariable, op1


def PatternOptimizer(p1, p2, ign=True):
    return OpKeyGraphRewriter(PatternNodeRewriter(p1, p2), ignore_newtrees=ign)


class TestFunction:
    @pytest.mark.xfail()
    def test_none(self):
        fn = function([], None)  # ok
        rval = fn()
        assert (
            rval != []
        ), "See #254: Using None as function output leads to [] return value"
        assert rval is None

    def test_empty(self):
        fn = function([], [])  # ok
        assert fn() == []

    def test_extra_inputs(self):
        x, s = scalars("xs")
        fn = function([x], [x])
        with pytest.raises(TypeError):
            fn(1, 2)

    def test_missing_inputs(self):
        def fn():
            x, s = scalars("xs")
            function([], [x])

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], [x], on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], [x])

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], x, on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], x)

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            # Ignore unused input s, as it hides the other error
            function([s], Out(x), on_unused_input="ignore")

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([s], Out(x))

        with pytest.raises(UnusedInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([In(x, update=s + x)], x)

        with pytest.raises(MissingInputError):
            fn()

        def fn():
            x, s = scalars("xs")
            function([In(x, update=((s * s) + x))], x)

        with pytest.raises(MissingInputError):
            fn()

    def test_input_anon_singleton(self):
        x, s = scalars("xs")
        fn = function([s, x], [x + s])
        assert fn(2, 3) == [5]
        # no state
        assert fn(2, 3) == [5]

    def test_input_anon_unpack(self):
        x, s = scalars("xs")
        fn = function([s, x], x + s)
        assert fn(2, 3) == 5

    def test_naming_rule0(self):
        x, s = scalars("xs")
        f = function([x, s], x / s)
        assert f(1, 2) == 0.5
        assert f(2, 1) == 2.0
        assert f(s=2, x=1) == 0.5
        assert f(x=2, s=1) == 2.0
        assert f(2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got multiple values for keyword argument 'x'
            f(2, x=2.0)
        with pytest.raises(TypeError):
            # takes exactly 2 non-keyword arguments (1 given)
            f(x=1)
        with pytest.raises(TypeError):
            # takes exactly 2 non-keyword arguments (0 given)
            f(s=1)

    def test_naming_rule1(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")
        f = function([a, s], a / s)
        assert f(1, 2) == 0.5
        assert f(2, 1) == 2.0
        assert f(2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got unexpected keyword argument 'q'
            f(q=2, s=1)
        with pytest.raises(TypeError):
            # got unexpected keyword argument 'a'
            f(a=2, s=1)

    def test_naming_rule2(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        # x's name is ignored because it is followed by anonymous parameter a.
        # Ignore unused input x, as it hides the other error
        f = function([x, a, s], a / s, on_unused_input="ignore")
        assert f(9, 1, 2) == 0.5
        assert f(9, 2, 1) == 2.0
        assert f(9, 2, s=1) == 2.0

        with pytest.raises(TypeError):
            # got unexpected keyword argument 'x'
            f(x=9, a=2, s=1)
        with pytest.raises(TypeError):
            # got unexpected keyword argument 'x'
            f(5.0, x=9)

    def test_naming_rule3(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        # x's name is not ignored (as in test_naming_rule2) because a has a default value.
        f = function([x, In(a, value=1.0), s], a / s + x)
        assert f(9, 2, 4) == 9.5  # can specify all args in order
        assert f(9, 2, s=4) == 9.5  # can give s as kwarg
        assert f(9, s=4) == 9.25  # can give s as kwarg, get default a
        assert f(x=9, s=4) == 9.25  # can give s as kwarg, omit a, x as kw
        with pytest.raises(TypeError):
            # got unexpected keyword argument 'a'
            f(x=9, a=2, s=4)
        with pytest.raises(TypeError):
            # takes exactly 3 non-keyword arguments (0 given)
            f()
        with pytest.raises(TypeError):
            # takes exactly 3 non-keyword arguments (1 given)
            f(x=9)

    def test_naming_rule4(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function([x, In(a, value=1.0, name="a"), s], a / s + x)

        assert f(9, 2, 4) == 9.5  # can specify all args in order
        assert f(9, 2, s=4) == 9.5  # can give s as kwarg
        assert f(9, s=4) == 9.25  # can give s as kwarg, get default a
        assert f(9, a=2, s=4) == 9.5  # can give s as kwarg, a as kwarg
        assert f(x=9, a=2, s=4) == 9.5  # can give all kwargs
        assert f(x=9, s=4) == 9.25  # can give all kwargs
        with pytest.raises(TypeError):
            # takes exactly 3 non-keyword arguments (0 given)
            f()
        with pytest.raises(TypeError):
            # got multiple values for keyword argument 'x'
            f(5.0, x=9)

    @pytest.mark.parametrize(
        "mode",
        [
            Mode(
                linker=VMLinker(allow_gc=True, use_cloop=False, c_thunks=False),
                optimizer="fast_compile",
            ),
            Mode(
                linker=VMLinker(allow_gc=True, use_cloop=False, c_thunks=False),
                optimizer="fast_run",
            ),
            Mode(linker="cvm", optimizer="fast_compile"),
            Mode(linker="cvm", optimizer="fast_run"),
        ],
    )
    def test_state_access(self, mode):
        a = scalar()
        x, s = scalars("xs")

        f = function(
            [x, In(a, value=1.0, name="a"), In(s, value=0.0, update=s + a * x)],
            s + a * x,
            mode=mode,
        )

        assert f[a] == 1.0
        assert f[s] == 0.0

        assert f(3.0) == 3.0
        assert f[s] == 3.0
        assert f(3.0, a=2.0) == 9.0  # 3.0 + 2*3.0

        assert (
            f[a] == 1.0
        )  # state hasn't changed permanently, we just overrode it last line
        assert f[s] == 9.0

        f[a] = 5.0
        assert f[a] == 5.0
        assert f(3.0) == 24.0  # 9 + 3*5
        assert f[s] == 24.0

    def test_same_names(self):
        a, x, s = scalars("xxx")
        # implicit names would cause error.  What do we do?
        f = function([a, x, s], a + x + s)
        assert f(1, 2, 3) == 6
        with pytest.raises(TypeError):
            f(1, 2, x=3)

    def test_weird_names(self):
        a, x, s = scalars("xxx")

        with pytest.raises(TypeError):
            function([In(a, name=[])], [])

        def t():
            f = function(
                [
                    In(a, name={"adsf", ()}, value=1.0),
                    In(x, name=(), value=2.0),
                    In(s, name=scalar(), value=3.0),
                ],
                a + x + s,
            )
            return f

        with pytest.raises(TypeError):
            t()

    def test_copy(self):
        a = scalar()
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )

        g = copy.copy(f)

        assert f.unpack_single == g.unpack_single
        assert f.trust_input == g.trust_input

        assert g.container[x].storage is not f.container[x].storage
        assert g.container[a].storage is not f.container[a].storage
        assert g.container[s].storage is not f.container[s].storage

        # Should not have been copied
        assert g.value[a] is f.value[a]

        # Should have been copied because it is mutable
        assert g.value[s] is not f.value[s]

        # Their contents should be equal, though
        assert np.array_equal(g.value[s], f.value[s])

        # They should be in sync, default value should be copied
        assert np.array_equal(f(2, 1), g(2))

        # They should be in sync, default value should be copied
        assert np.array_equal(f(2, 1), g(2))

        # Put them out of sync
        f(1, 2)

        # They should not be equal anymore
        assert not np.array_equal(f(1, 2), g(1, 2))

    def test_copy_share_memory(self):
        x = fscalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1)
        z = shared(value=2)
        out = tanh((x + y + 2) / (x + z - 0.2) ** 2)

        # Test for different linkers
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], [out], mode=mode, updates={z: z + 1})
            cpy = ori.copy(share_memory=True)

            # Test if memories shared
            storage_map_ori = ori.vm.storage_map
            storage_map_cpy = cpy.vm.storage_map
            fgraph_cpy = cpy.maker.fgraph

            # Assert intermediate and Constants storages are shared.
            # and output stoarges are not shared
            i_o_variables = fgraph_cpy.inputs + fgraph_cpy.outputs
            ori_storages = storage_map_ori.values()
            l = [
                val
                for key, val in storage_map_cpy.items()
                if key not in i_o_variables or isinstance(key, Constant)
            ]
            for storage in l:
                assert any(storage is s for s in ori_storages)

            # Assert storages of SharedVariable without updates are shared
            for (input, _1, _2), here, there in zip(
                ori.indices, ori.input_storage, cpy.input_storage
            ):
                assert here.data is there.data

    def test_swap_SharedVariable(self):
        i = iscalar()
        x_list = shared(value=np.random.random((10,)).astype(config.floatX))

        x = scalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1, name="y")
        z = shared(value=2, name="z")
        m = shared(value=0, name="m")

        # SharedVariable to replace
        y_rpl = shared(value=3, name="y_rpl")
        z_rpl = shared(value=4, name="z_rpl")
        swap = {y: y_rpl, z: z_rpl}
        map_SV = {"y_rpl": y_rpl, "z_rpl": z_rpl}

        out = x + y + z + m

        # Test for different linkers
        # for mode in ["FAST_RUN","FAST_COMPILE"]:
        second_time = False
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function(
                [i],
                [out],
                mode=mode,
                updates=[(z, z + 1), (m, m + 2)],
                givens={x: x_list[i]},
            )
            cpy = ori.copy(swap=swap)

            # run function several times
            ori(1), cpy(1), cpy(2)

            # assert same SharedVariable are update in different function
            if not second_time:
                # m should be updated 3 times
                assert m.get_value() == 6
                # z should be updated once
                assert z.get_value() == 3
                # z_rpl should be updated twice
                assert z_rpl.get_value() == 6
                # y and y_rpl should not be updated
                assert y_rpl.get_value() == 3
                assert y.get_value() == 1
            elif second_time:
                # doule update for sharedvariable
                assert m.get_value() == 12
                assert z.get_value() == 4
                assert z_rpl.get_value() == 8
                assert y_rpl.get_value() == 3

            # test cpy function:
            # 2. SharedVariable is updatable -> values did update(z == 5)
            # 1. sharedvariable is swap ->  Rpl sharedvariables share storage
            names = map_SV.keys()
            for key in cpy.vm.storage_map:
                if key.name in names:
                    assert (
                        map_SV[key.name].container.storage[0]
                        == cpy.vm.storage_map[key][0]
                    )

            second_time = True

    def test_swap_SharedVariable_with_given(self):
        # A special testcase for logistic_sgd.py in Deep Learning Tutorial
        # This test assert that SharedVariable in different function have same storage

        train_x = shared(value=np.random.random((10, 10)).astype(config.floatX))
        test_x = shared(value=np.random.random((10, 10)).astype(config.floatX))

        train_y = shared(value=np.random.random((10, 1)).astype(config.floatX))
        test_y = shared(value=np.random.random((10, 1)).astype(config.floatX))

        i = iscalar("index")
        x = vector("x")
        y = vector("y")
        # this formular has no sense but for a test
        out = (at_sum(x) - y) ** 2
        train = function(
            [i],
            out,
            givens={x: train_x[i], y: train_y[i]},
            updates={train_x: train_x + 0.1},
        )

        test_def = function([i], out, givens={x: test_x[i], y: test_y[i]})
        test_cpy = train.copy(
            swap={train_x: test_x, train_y: test_y}, delete_updates=True
        )

        for in1, in2 in zip(test_def.maker.inputs, test_cpy.maker.inputs):
            assert in1.value is in2.value

    def test_copy_delete_updates(self):
        w = iscalar("w")
        x = fscalar("x")
        # SharedVariable for tests, one of them has update
        y = shared(value=1, name="y")
        z = shared(value=2, name="z")
        out = x + y + z

        # Test for different linkers
        # for mode in ["FAST_RUN","FAST_COMPILE"]:
        # second_time = False
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], out, mode=mode, updates={z: z * 2})
            cpy = ori.copy(delete_updates=True)

            assert cpy(1) == 4
            assert cpy(1) == 4
            assert cpy(1) == 4

        # Test if unused implicit and explicit inputs from delete_updates
        # are ignored as intended.
        for mode in ("FAST_RUN", "FAST_COMPILE"):
            ori = function([x], x, mode=mode, updates={z: z * 2})
            cpy = ori.copy(delete_updates=True)

            ori = function([x, w], x, mode=mode, updates={z: z + w})
            cpy = ori.copy(delete_updates=True)

    def test_shared_state0(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        g = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=f.container[s], update=s - a * x, mutable=True),
            ],
            s + a * x,
        )

        f(1, 2)
        assert f[s] == 2
        assert g[s] == 2
        g(1, 2)
        assert f[s] == 0
        assert g[s] == 0

    def test_shared_state1(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        g = function(
            [x, In(a, value=1.0, name="a"), In(s, value=f.container[s])], s + a * x
        )

        f(1, 2)
        assert f[s] == 2
        assert g[s] == 2
        f(1, 2)
        g(1, 2)
        assert f[s] == 4
        assert g[s] == 4

    def test_shared_state2(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=False),
            ],
            s + a * x,
        )
        g = function(
            [x, In(a, value=1.0, name="a"), In(s, value=f.container[s])], s + a * x
        )

        f(1, 2)
        assert f[s] == 2
        assert g[s] == 2
        f(1, 2)
        assert f[s] == 4
        assert g[s] == 4
        g(1, 2)  # has no effect on state
        assert f[s] == 4
        assert g[s] == 4

    def test_shared_state_not_implicit(self):
        # This test is taken from the documentation in
        # doc/topics/function.txt. If it does not pass anymore and yet the
        # behavior is still intended the doc and the test should both be
        # updated accordingly.
        x, s = scalars("xs")
        inc = function([x, In(s, update=(s + x), value=10.0)], [])
        dec = function(
            [x, In(s, update=(s - x), value=inc.container[s], implicit=False)], []
        )
        assert dec[s] is inc[s]
        inc[s] = 2
        assert dec[s] == 2
        dec(1)
        assert inc[s] == 1
        dec(1, 0)
        assert inc[s] == -1
        assert dec[s] == -1

    def test_constant_output(self):
        # Test that if the output is a constant, we respect the aesara memory interface
        f = function([], at.constant([4]))
        # print f.maker.fgraph.toposort()
        out = f()
        assert (out == 4).all()
        out[0] = 3
        out2 = f()
        # If the following 2 asserts fail it mean Aesara broke it's memory contract.
        assert out2 is not out
        assert (out2 == 4).all()

        # Test that if the output is a constant and borrow, we respect the aesara memory interface
        f = function([], Out(at.constant([4]), borrow=True))
        # print f.maker.fgraph.toposort()
        out = f()
        assert (out == 4).all()
        out[0] = 3
        out2 = f()

        if isinstance(get_default_mode(), DebugMode):
            # In DebugMode, we don't implement optimization based on borrow on the output.
            assert (out2 == 4).all()
        else:
            assert out2 is out
            assert (out2 == 3).all()

    def test_borrow_input(self):
        # Tests that the contract for io.In is respected. When borrow=False, it should be
        # impossible for outputs to be aliased to the input variables provided by the user,
        # either through a view-map or a destroy map. New tests should be added in the future
        # when borrow=True is implemented.

        a = dmatrix()
        aval = np.random.random((3, 3))

        # when borrow=False, test that a destroy map cannot alias output to input
        f = function([In(a, borrow=False)], Out(a + 1, borrow=True))
        assert np.all(f(aval) == aval + 1)
        assert not np.may_share_memory(aval, f(aval))

        # when borrow=False, test that a viewmap cannot alias output to input
        f = function([In(a, borrow=False)], Out(a[0, :], borrow=True))
        assert np.all(f(aval) == aval[0, :])
        assert not np.may_share_memory(aval, f(aval))

    def test_borrow_output(self):
        a = dmatrix()
        f = function([a], Out(a, borrow=False))
        o = np.ones((3, 3))
        assert o is not f(o)  # function no longer permits aliasing outputs to inputs

        f = function([a], Out(a * 4, borrow=False))
        o = np.ones((3, 3))
        four = f(o)
        assert np.all(four == 4)
        f(o + 0.1)  # should not clobber the memory used to store four
        assert np.all(four == 4)

        f = function([a], Out(a * 4, borrow=True), mode=Mode("c|py_nogc", "fast_run"))
        o = np.ones((3, 3))
        four = f(o)
        assert np.all(four == 4)
        f(o + 0.1)  # should clobber the memory used to store four
        if config.cxx:
            assert not np.all(four == 4)
        else:
            # The Elemwise.perform method don't reuse memory
            # as some numpy version don't support that correctly.
            assert np.all(four == 4)

    def test_disconnected_input(self):
        a = scalar("a")
        v = vector("v")
        with pytest.raises(UnusedInputError):
            function([a, v], v * 2)

        function([a, v], v * 2, on_unused_input="ignore")

    def test_masked_input(self):
        m = matrix("m")
        mt = m.T
        mt.name = "m.T"
        with pytest.raises(UnusedInputError):
            function([m, mt], mt * 2)
        function([m, mt], mt * 2, on_unused_input="ignore")

    def test_givens_input_var(self):
        # Ensure error is raised when trying to replace an input variable.

        x = scalar("x")
        y = x * 2
        with pytest.raises(RuntimeError):
            function([x], y, givens={x: x + 1})

    def test_free(self):
        # Make test on free() function

        x = vector("x")
        func = function([x], x + 1)
        func.vm.allow_gc = False
        func([1])

        check_list = []
        for key, val in func.vm.storage_map.items():
            if not isinstance(key, Constant):
                check_list.append(val)
        assert any(val[0] for val in check_list)

        func.free()

        for key, val in func.vm.storage_map.items():
            if not isinstance(key, Constant):
                assert val[0] is None

    def test_default_values(self):
        # Check that default values are restored
        # when an exception occurs in interactive mode.

        a, b = dscalars("a", "b")
        c = a + b
        funct = function([In(a, name="first"), In(b, value=1, name="second")], c)
        x = funct(first=1)
        try:
            funct(second=2)
        except TypeError:
            assert funct(first=1) == x

    def test_check_for_aliased_inputs(self):
        b = np.random.random((5, 4))
        s1 = shared(b)
        s2 = shared(b)
        x1 = vector()

        # Assert cases we should not check for aliased inputs
        for d in [
            dict(outputs=[s1 + 1]),
            dict(outputs=[s1 + 1, s2 + 3]),
            dict(outputs=[s1 + 1], updates=[(s2, s2 + 3)]),
            dict(inputs=[x1], outputs=[x1 + 1], updates=[(s2, s2 + 3)]),
        ]:
            if "inputs" not in d:
                d["inputs"] = []
            f = function(**d)
            assert not f._check_for_aliased_inputs, d

        # Assert cases we should check for aliased inputs
        for d in [
            dict(
                inputs=[In(x1, borrow=True)],
                outputs=[x1 + 1],
                updates=[(s2, s2 + 3)],
            ),
            dict(
                inputs=[In(x1, borrow=True, mutable=True)],
                outputs=[x1 + 1],
                updates=[(s2, s2 + 3)],
            ),
            dict(
                inputs=[In(x1, mutable=True)],
                outputs=[x1 + 1],
                updates=[(s2, s2 + 3)],
            ),
        ]:
            if "inputs" not in d:
                d["inputs"] = []
            f = function(**d)

            assert f._check_for_aliased_inputs, d

    def test_output_dictionary(self):
        # Tests that function works when outputs is a dictionary

        x = scalar()
        f = function([x], outputs={"a": x, "c": x * 2, "b": x * 3, "1": x * 4})

        outputs = f(10.0)

        assert outputs["a"] == 10.0
        assert outputs["b"] == 30.0
        assert outputs["1"] == 40.0
        assert outputs["c"] == 20.0

    def test_input_named_variables(self):
        # Tests that named variables work when outputs is a dictionary

        x = scalar("x")
        y = scalar("y")

        f = function([x, y], outputs={"a": x + y, "b": x * y})

        assert f(2, 4) == {"a": 6, "b": 8}
        assert f(2, y=4) == f(2, 4)
        assert f(x=2, y=4) == f(2, 4)

    def test_output_order_sorted(self):
        # Tests that the output keys are sorted correctly.

        x = scalar("x")
        y = scalar("y")
        z = scalar("z")
        e1 = scalar("1")
        e2 = scalar("2")

        f = function(
            [x, y, z, e1, e2], outputs={"x": x, "y": y, "z": z, "1": e1, "2": e2}
        )

        assert "1" in str(f.outputs[0])
        assert "2" in str(f.outputs[1])
        assert "x" in str(f.outputs[2])
        assert "y" in str(f.outputs[3])
        assert "z" in str(f.outputs[4])

    def test_composing_function(self):
        # Tests that one can compose two aesara functions when the outputs are
        # provided in a dictionary.

        x = scalar("x")
        y = scalar("y")

        a = x + y
        b = x * y

        f = function([x, y], outputs={"a": a, "b": b})

        a = scalar("a")
        b = scalar("b")

        l = a + b
        r = a * b

        g = function([a, b], outputs=[l, r])

        result = g(**f(5, 7))

        assert result[0] == 47.0
        assert result[1] == 420.0

    def test_output_list_still_works(self):
        # Test that function works if outputs is a list.

        x = scalar("x")

        f = function([x], outputs=[x * 3, x * 2, x * 4, x])

        result = f(5.0)

        assert result[0] == 15.0
        assert result[1] == 10.0
        assert result[2] == 20.0
        assert result[3] == 5.0

    def test_key_string_requirement(self):
        # Tests that an exception is thrown if a non-string key is used in
        # the outputs dictionary.
        x = scalar("x")

        with pytest.raises(AssertionError):
            function([x], outputs={1.0: x})

        with pytest.raises(AssertionError):
            function([x], outputs={1.0: x, "a": x**2})

        with pytest.raises(AssertionError):
            function([x], outputs={(1, "b"): x, 1.0: x**2})


class TestPicklefunction:
    def test_deepcopy(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        try:
            g = copy.deepcopy(f)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        # if they both return, assume  that they return equivalent things.
        # print [(k,id(k)) for k in f.finder.keys()]
        # print [(k,id(k)) for k in g.finder.keys()]

        assert g.container[0].storage is not f.container[0].storage
        assert g.container[1].storage is not f.container[1].storage
        assert g.container[2].storage is not f.container[2].storage
        assert x not in g.container
        assert x not in g.value
        assert len(f.defaults) == len(g.defaults)
        assert f._check_for_aliased_inputs is g._check_for_aliased_inputs
        assert f.name == g.name
        assert f.maker.fgraph.name == g.maker.fgraph.name
        # print 'f.defaults = %s' % (f.defaults, )
        # print 'g.defaults = %s' % (g.defaults, )
        for (f_req, f_feed, f_val), (g_req, g_feed, g_val) in zip(
            f.defaults, g.defaults
        ):
            assert f_req == g_req and f_feed == g_feed and f_val == g_val

        assert g.value[1] is not f.value[1]  # should not have been copied
        assert (
            g.value[2] is not f.value[2]
        )  # should have been copied because it is mutable.
        assert not (g.value[2] != f.value[2]).any()  # its contents should be identical

        assert f(2, 1) == g(
            2
        )  # they should be in sync, default value should be copied.
        assert f(2, 1) == g(
            2
        )  # they should be in sync, default value should be copied.
        f(1, 2)  # put them out of sync
        assert f(1, 2) != g(1, 2)  # they should not be equal anymore.
        g(1, 2)  # put them back in sync
        assert f(3) == g(3)  # They should be in sync again.

    def test_deepcopy_trust_input(self):
        a = dscalar()  # the a is for 'anonymous' (un-named).
        x, s = dscalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        f.trust_input = True
        try:
            g = copy.deepcopy(f)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        assert f.trust_input is g.trust_input
        f(np.asarray(2.0))
        with pytest.raises((ValueError, AttributeError, InvalidValueError)):
            f(2.0)
        g(np.asarray(2.0))
        with pytest.raises((ValueError, AttributeError, InvalidValueError)):
            g(2.0)

    def test_output_keys(self):
        x = vector()
        f = function([x], {"vec": x**2})
        o = f([2, 3, 4])
        assert isinstance(o, dict)
        assert np.allclose(o["vec"], [4, 9, 16])
        g = copy.deepcopy(f)
        o = g([2, 3, 4])
        assert isinstance(o, dict)
        assert np.allclose(o["vec"], [4, 9, 16])

    def test_deepcopy_shared_container(self):
        # Ensure that shared containers remain shared after a deep copy.
        a, x = scalars("ax")

        h = function([In(a, value=0.0)], a)
        f = function([x, In(a, value=h.container[a], implicit=True)], x + a)

        try:
            memo = {}
            ac = copy.deepcopy(a)
            memo.update({id(a): ac})
            hc = copy.deepcopy(h, memo=memo)
            memo.update({id(h): hc})
            fc = copy.deepcopy(f, memo=memo)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        h[a] = 1
        hc[ac] = 2
        assert f[a] == 1
        assert fc[ac] == 2

    def test_pickle(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")

        f = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )

        try:
            # Note that here we also test protocol 0 on purpose, since it
            # should work (even though one should not use it).
            g = pickle.loads(pickle.dumps(f, protocol=0))
            g = pickle.loads(pickle.dumps(f, protocol=-1))
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise
        # if they both return, assume  that they return equivalent things.
        # print [(k,id(k)) for k in f.finder.keys()]
        # print [(k,id(k)) for k in g.finder.keys()]

        assert g.container[0].storage is not f.container[0].storage
        assert g.container[1].storage is not f.container[1].storage
        assert g.container[2].storage is not f.container[2].storage
        assert x not in g.container
        assert x not in g.value

        assert g.value[1] is not f.value[1]  # should not have been copied
        assert (
            g.value[2] is not f.value[2]
        )  # should have been copied because it is mutable.
        assert not (g.value[2] != f.value[2]).any()  # its contents should be identical

        assert f(2, 1) == g(
            2
        )  # they should be in sync, default value should be copied.
        assert f(2, 1) == g(
            2
        )  # they should be in sync, default value should be copied.
        f(1, 2)  # put them out of sync
        assert f(1, 2) != g(1, 2)  # they should not be equal anymore.

    def test_optimizations_preserved(self):
        a = dvector()  # the a is for 'anonymous' (un-named).
        x = dvector("x")
        s = dvector("s")
        xm = dmatrix("x")
        sm = dmatrix("s")

        f = function(
            [a, x, s, xm, sm],
            ((a.T.T) * (dot(xm, (sm.T.T.T)) + x).T * (x / x) + s),
        )
        old_default_mode = config.mode
        old_default_opt = config.optimizer
        old_default_link = config.linker
        try:
            try:
                str_f = pickle.dumps(f, protocol=-1)
                config.mode = "Mode"
                config.linker = "py"
                config.optimizer = "None"
                g = pickle.loads(str_f)
                # print g.maker.mode
                # print compile.mode.default_mode
            except NotImplementedError as e:
                if e[0].startswith("DebugMode is not pickl"):
                    g = "ok"
        finally:
            config.mode = old_default_mode
            config.optimizer = old_default_opt
            config.linker = old_default_link

        if g == "ok":
            return

        assert f.maker is not g.maker
        assert f.maker.fgraph is not g.maker.fgraph
        tf = f.maker.fgraph.toposort()
        tg = f.maker.fgraph.toposort()
        assert len(tf) == len(tg)
        for nf, ng in zip(tf, tg):
            assert nf.op == ng.op
            assert len(nf.inputs) == len(ng.inputs)
            assert len(nf.outputs) == len(ng.outputs)
            assert [i.type for i in nf.inputs] == [i.type for i in ng.inputs]
            assert [i.type for i in nf.outputs] == [i.type for i in ng.outputs]

    def test_multiple_functions(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")
        v = vector("v")

        # put in some inputs
        list_of_things = [s, x, v]

        # some derived thing, whose inputs aren't all in the list
        list_of_things.append(a * x + s)

        f1 = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        list_of_things.append(f1)

        # now put in a function sharing container with the previous one
        f2 = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=f1.container[s], update=s + a * x, mutable=True),
            ],
            s + a * x,
        )
        list_of_things.append(f2)

        assert isinstance(f2.container[s].storage, list)
        assert f2.container[s].storage is f1.container[s].storage

        # now put in a function with non-scalar
        v_value = np.asarray([2, 3, 4.0], dtype=config.floatX)
        f3 = function([x, In(v, value=v_value)], x + v)
        list_of_things.append(f3)

        # try to pickle the entire things
        try:
            saved_format = pickle.dumps(list_of_things, protocol=-1)
            new_list_of_things = pickle.loads(saved_format)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise

        # now test our recovered new_list_of_things
        # it should be totally unrelated to the original
        # it should be interdependent in the same way as the original

        ol = list_of_things
        nl = new_list_of_things

        for i in range(4):
            assert nl[i] != ol[i]
            assert nl[i].type == ol[i].type
            assert nl[i].type is not ol[i].type

        # see if the implicit input got stored
        assert ol[3].owner.inputs[1] is s
        assert nl[3].owner.inputs[1] is not s
        assert nl[3].owner.inputs[1].type == s.type

        # moving on to the functions...
        for i in range(4, 7):
            assert nl[i] != ol[i]

        # looking at function number 1, input 's'
        assert nl[4][nl[0]] is not ol[4][ol[0]]
        assert nl[4][nl[0]] == ol[4][ol[0]]
        assert nl[4](3) == ol[4](3)

        # looking at function number 2, input 's'
        # make sure it's shared with the first function
        assert ol[4].container[ol[0]].storage is ol[5].container[ol[0]].storage
        assert nl[4].container[nl[0]].storage is nl[5].container[nl[0]].storage
        assert nl[5](3) == ol[5](3)
        assert nl[4].value[nl[0]] == 6

        assert np.all(nl[6][nl[2]] == np.asarray([2, 3.0, 4]))

    def test_broken_pickle_with_shared(self):
        saves = []

        def pers_save(obj):
            if isinstance(obj, np.ndarray):
                saves.append(obj)
                return len(saves) - 1
            else:
                return None

        def pers_load(id):
            return saves[id]

        b = np.random.random((5, 4))

        x = matrix()
        y = shared(b)

        f = function([x], dot(x, y))

        from io import BytesIO

        fp = BytesIO()
        p = pickle.Pickler(fp, 2)
        p.persistent_id = pers_save
        try:
            p.dump(f)
        except NotImplementedError as e:
            if exc_message(e).startswith("DebugMode is not picklable"):
                return
            else:
                raise
        fp2 = BytesIO(fp.getvalue())
        fp.close()
        p = pickle.Unpickler(fp2)
        p.persistent_load = pers_load
        p.load()
        fp2.close()

    def test_pickle_class_with_functions(self):
        blah = SomethingToPickle()
        assert blah.f2.container[blah.s].storage is blah.f1.container[blah.s].storage

        try:
            blah2 = copy.deepcopy(blah)
        except NotImplementedError as e:
            if e[0].startswith("DebugMode is not picklable"):
                return
            else:
                raise

        assert (
            blah2.f2.container[blah2.s].storage is blah2.f1.container[blah2.s].storage
        )

        assert blah.f1[blah.s] == blah2.f1[blah2.s]

        blah.f2(5)
        assert blah.f1[blah.s] != blah2.f1[blah2.s]


class SomethingToPickle:
    def __init__(self):
        a = scalar()  # the a is for 'anonymous' (un-named).
        x, s = scalars("xs")
        v = vector("v")

        self.s = s
        self.x = x
        self.v = v

        self.e = a * x + s

        self.f1 = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=0.0, update=s + a * x, mutable=True),
            ],
            s + a * x,
        )

        self.f2 = function(
            [
                x,
                In(a, value=1.0, name="a"),
                In(s, value=self.f1.container[s], update=s + a * x, mutable=True),
            ],
            s + a * x,
        )


def test_empty_givens_updates():
    # Regression test for bug fixed in 8625e03.

    # Empty givens / updates dictionaries were not properly detected before,
    # triggering useless crashes at compile time.
    x = scalar()
    y = x * 2
    function([In(x)], y, givens={})
    function([In(x)], y, updates={})


def test_update_placeholder():
    a, x, s, m, n = scalars("axsmn")

    f1 = function(
        [
            x,
            In(a, value=1.0, name="a"),
            In(m, value=0.0, update=update_placeholder(m), mutable=True),
            In(s, value=0.0, update=s + a * x, mutable=True),
            In(n, value=0.0, update=update_placeholder(n), mutable=True),
        ],
        s + a * x,
    )

    # The second update shouldn't be present
    assert len(f1.maker.fgraph.outputs) == 2
    assert f1.maker.fgraph.update_mapping == {1: 3}


class TestSupervisor:
    def test_basic(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        hf = Supervisor([var1])
        fg.attach_feature(hf)

        assert fg._supervisor_protected == {var1}

        # Make sure we can update the protected variables by
        # adding another `Supervisor`
        hf = Supervisor([var2])
        fg.attach_feature(hf)

        assert fg._supervisor_protected == {var1, var2}

    def test_pickle(self):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        hf = Supervisor([var1])
        fg.attach_feature(hf)

        fg_pkld = pickle.dumps(fg)
        fg_unpkld = pickle.loads(fg_pkld)

        assert any(isinstance(ft, Supervisor) for ft in fg_unpkld._features)
        assert all(hasattr(fg, attr) for attr in ("_supervisor_protected",))

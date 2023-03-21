import time

import numpy as np
import pytest

from aesara.compile.function import function
from aesara.compile.io import In
from aesara.compile.mode import Mode, get_mode
from aesara.compile.sharedvalue import shared
from aesara.configdefaults import config
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.ifelse import ifelse
from aesara.link.c.basic import OpWiseCLinker
from aesara.link.c.exceptions import MissingGXX
from aesara.link.utils import map_storage
from aesara.link.vm import VM, Loop, Stack, VMLinker
from aesara.tensor.math import cosh, tanh
from aesara.tensor.type import lscalar, scalar, scalars, vector, vectors
from aesara.tensor.var import TensorConstant
from tests import unittest_tools as utt


class SomeOp(Op):
    def perform(self, node, inputs, outputs):
        pass

    def make_node(self, x):
        return Apply(self, [x], [x.type()])


class TestCallbacks:
    # Test the `VMLinker`'s callback argument, which can be useful for debugging.

    def setup_method(self):
        self.n_callbacks = {}

    def callback(self, node, thunk, storage_map, compute_map):
        key = node.op.__class__.__name__
        self.n_callbacks.setdefault(key, 0)
        self.n_callbacks[key] += 1

    def test_callback(self):
        a, b, c = scalars("abc")
        f = function(
            [a, b, c],
            (a + b) + c,
            mode=Mode(optimizer=None, linker=VMLinker(callback=self.callback)),
        )

        f(1, 2, 3)
        assert sum(self.n_callbacks.values()) == len(f.maker.fgraph.toposort())
        f(1, 2, 3)
        assert sum(self.n_callbacks.values()) == len(f.maker.fgraph.toposort()) * 2

    def test_callback_with_ifelse(self):
        a, b, c = scalars("abc")
        f = function(
            [a, b, c],
            ifelse(a, 2 * b, 2 * c),
            mode=Mode(optimizer=None, linker=VMLinker(callback=self.callback)),
        )

        f(1, 2, 3)
        assert self.n_callbacks["IfElse"] == 2


def test_use_c_thunks():
    a_at = scalars("a")
    b_at = vectors("b")

    a = np.array(0.0).astype(config.floatX)
    b = np.array([2.0]).astype(config.floatX)

    cases = [False]
    if config.cxx:
        cases.append(True)

    for use_c_thunks in cases:
        f = function(
            [a_at, b_at],
            a_at * b_at,
            mode=Mode(
                optimizer=None, linker=VMLinker(c_thunks=use_c_thunks, use_cloop=False)
            ),
        )
        assert np.array_equal(a * b, f(a, b))
        assert any(hasattr(t, "cthunk") for t in f.vm.thunks) == use_c_thunks


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_speed():
    # TODO FIXME: This isn't a real test.

    def build_graph(x, depth=5):
        z = x
        for d in range(depth):
            z = z + z
        return z

    def numpy_version(x, depth):
        z = x
        for d in range(depth):
            z = z + z
        return z

    def time_numpy():
        steps_a = 5
        steps_b = 100
        x = np.asarray([2.0, 3.0], dtype=config.floatX)

        numpy_version(x, steps_a)
        t0 = time.perf_counter()
        # print numpy_version(x, steps_a)
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        # print numpy_version(x, steps_b)
        t3 = time.perf_counter()
        t_a = t1 - t0
        t_b = t3 - t2

        print(f"numpy takes {1000 * (t_b - t_a) / (steps_b - steps_a):f} s/Kop")

    def time_linker(name, linker):
        steps_a = 5
        steps_b = 100
        x = vector()
        a = build_graph(x, steps_a)
        b = build_graph(x, steps_b)

        f_a = function([x], a, mode=Mode(optimizer=None, linker=linker()))
        f_b = function([x], b, mode=Mode(optimizer=None, linker=linker()))

        f_a([2.0, 3.0])
        t0 = time.perf_counter()
        f_a([2.0, 3.0])
        t1 = time.perf_counter()

        f_b([2.0, 3.0])

        t2 = time.perf_counter()
        f_b([2.0, 3.0])
        t3 = time.perf_counter()

        t_a = t1 - t0
        t_b = t3 - t2

        print(f"{name} takes {1000 * (t_b - t_a) / (steps_b - steps_a):f} s/Kop")

    time_linker("c|py", OpWiseCLinker)
    time_linker("vmLinker", VMLinker)
    time_linker("vmLinker_nogc", lambda: VMLinker(allow_gc=False))
    if config.cxx:
        time_linker("vmLinker_CLOOP", lambda: VMLinker(allow_gc=False, use_cloop=True))
    time_numpy()


@pytest.mark.parametrize(
    "linker",
    [
        VMLinker(),
        VMLinker(allow_gc=False),
        VMLinker(allow_gc=False, use_cloop=True),
    ],
)
def test_speed_lazy(linker):
    # TODO FIXME: This isn't a real test.

    def build_graph(x, depth=5):
        z = x
        for d in range(depth):
            z = ifelse(z[0] > 0, -z, z)
        return z

    steps_a = 10
    steps_b = 100
    x = vector()
    a = build_graph(x, steps_a)
    b = build_graph(x, steps_b)

    f_a = function([x], a, mode=Mode(optimizer=None, linker=linker))
    f_b = function([x], b, mode=Mode(optimizer=None, linker=linker))

    f_a([2.0])
    t0 = time.perf_counter()
    f_a([2.0])
    t1 = time.perf_counter()

    f_b([2.0])

    t2 = time.perf_counter()
    f_b([2.0])
    t3 = time.perf_counter()

    t_a = t1 - t0
    t_b = t3 - t2

    print(f"{linker} takes {1000 * (t_b - t_a) / (steps_b - steps_a):f} s/Kop")


@pytest.mark.parametrize(
    "linker", [VMLinker(allow_partial_eval=True, use_cloop=False), "cvm"]
)
def test_partial_function(linker):
    x = scalar("input")
    y = x**2
    f = function(
        [x], [y + 7, y - 9, y / 14.0], mode=Mode(optimizer=None, linker=linker)
    )

    if linker == "cvm":
        from aesara.link.c.cvm import CVM

        assert isinstance(f.vm, CVM)
    else:
        assert isinstance(f.vm, Stack)

    assert f(3, output_subset=[0, 1, 2]) == f(3)
    assert f(4, output_subset=[0, 2]) == [f(4)[0], f(4)[2]]

    utt.assert_allclose(f(5), np.array([32.0, 16.0, 1.7857142857142858]))


@pytest.mark.parametrize(
    "linker", [VMLinker(allow_partial_eval=True, use_cloop=False), "cvm"]
)
def test_partial_function_with_output_keys(linker):
    x = scalar("input")
    y = 3 * x
    f = function(
        [x], {"a": y * 5, "b": y - 7}, mode=Mode(optimizer=None, linker=linker)
    )

    assert f(5, output_subset=["a"])["a"] == f(5)["a"]


@pytest.mark.parametrize(
    "linker", [VMLinker(allow_partial_eval=True, use_cloop=False), "cvm"]
)
def test_partial_function_with_updates(linker):
    x = lscalar("input")
    y = shared(np.asarray(1, "int64"), name="global")

    mode = Mode(optimizer=None, linker=linker)

    f = function(
        [x],
        [x, x + 34],
        updates=[(y, x + 1)],
        mode=mode,
    )
    g = function(
        [x],
        [x - 6],
        updates=[(y, y + 3)],
        mode=mode,
    )

    assert f(3, output_subset=[]) == []
    assert y.get_value() == 4
    assert g(30, output_subset=[0]) == [24]
    assert g(40, output_subset=[]) == []
    assert y.get_value() == 10


def test_allow_gc_cvm():
    mode = config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"

    v = vector()
    f = function([v], v + 1, mode=mode)

    f([1])
    n = list(f.maker.fgraph.apply_nodes)[0].outputs[0]
    assert f.vm.storage_map[n][0] is None
    assert f.vm.allow_gc is True

    f.vm.allow_gc = False
    assert f.vm.allow_gc is False
    f([1])
    assert f.vm.storage_map[n][0] is not None
    f.vm.allow_gc = True
    assert f.vm.allow_gc is True
    f([1])
    assert f.vm.storage_map[n][0] is None


class RunOnce(Op):
    __props__ = ("nb_run",)

    def __init__(self):
        self.nb_run = 0

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        assert self.nb_run == 0
        self.nb_run += 1
        outputs[0][0] = inputs[0].copy()


def test_vm_gc():
    x = vector()
    p = RunOnce()(x)
    mode = Mode(linker=VMLinker(lazy=True))
    f = function([In(x, mutable=True)], [p + 1, p + 2], mode=mode)
    f([1, 2, 3])

    p = RunOnce()(x)
    pp = p + p
    f = function([x], [pp + pp], mode=mode)
    f([1, 2, 3])


def test_reallocation():
    x = scalar("x")
    y = scalar("y")
    z = tanh(3 * x + y) + cosh(x + 5 * y)
    # The functionality is currently implement for non lazy and non c VM only.
    for linker in [
        VMLinker(allow_gc=False, lazy=False, use_cloop=False),
        VMLinker(allow_gc=True, lazy=False, use_cloop=False),
    ]:
        m = get_mode(Mode(linker=linker))
        m = m.excluding("fusion", "inplace")

        f = function([x, y], z, name="test_reduce_memory", mode=m)
        output = f(1, 2)
        assert output
        storage_map = f.vm.storage_map

        def check_storage(storage_map):
            for i in storage_map:
                if not isinstance(i, TensorConstant):
                    keys_copy = list(storage_map.keys())[:]
                    keys_copy.remove(i)
                    for o in keys_copy:
                        if storage_map[i][0] and storage_map[i][0] is storage_map[o][0]:
                            return [True, storage_map[o][0]]
            return [False, None]

        assert check_storage(storage_map)[0]
        assert len({id(v) for v in storage_map.values()}) < len(storage_map)


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_no_recycling():
    x = vector()
    for lnk in [
        VMLinker(use_cloop=True),
        VMLinker(use_cloop=False, lazy=True),
        VMLinker(use_cloop=False, lazy=False, allow_gc=True),
        VMLinker(use_cloop=False, lazy=False, allow_gc=False),
    ]:
        mode = Mode(optimizer="fast_compile", linker=lnk)
        f = function([x], x + 1, mode=mode)
        f2 = function([x], (x + 1) * 2, mode=mode)
        m1 = f.vm.thunks[0].thunk.module
        m2 = f2.vm.thunks[0].thunk.module
        assert m1 is m2


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_VMLinker_make_vm_cvm():
    # We don't want this at module level, since CXX might not be present
    from aesara.link.c.cvm import CVM

    a = scalar()
    linker = VMLinker(allow_gc=False, use_cloop=True)

    f = function([a], a, mode=Mode(optimizer=None, linker=linker))
    assert isinstance(f.vm, CVM)


def test_VMLinker_make_vm_no_cvm():
    from importlib import reload
    from unittest.mock import patch

    with config.change_flags(cxx=""):
        # Make sure that GXX isn't present
        with pytest.raises(MissingGXX):
            import aesara.link.c.cvm

            reload(aesara.link.c.cvm)

        # Make sure that `cvm` module is missing
        with patch.dict("sys.modules", {"aesara.link.c.cvm": None}):
            a = scalar()
            linker = VMLinker(allow_gc=False, use_cloop=True)

            with pytest.raises(ModuleNotFoundError):
                import aesara.link.c.cvm

            f = function([a], a, mode=Mode(optimizer=None, linker=linker))
            assert isinstance(f.vm, Loop)


def test_VMLinker_exception():
    class BadOp(Op):
        def perform(self, node, inputs, outputs):
            pass

        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def make_thunk(self, *args, **kwargs):
            raise Exception("bad Op")

    a = scalar()
    linker = VMLinker(allow_gc=False, use_cloop=True)

    z = BadOp()(a)

    with pytest.raises(Exception, match=".*Apply node that caused the error.*"):
        function([a], z, mode=Mode(optimizer=None, linker=linker))


def test_VM_exception():
    class SomeVM(VM):
        def __call__(self):
            pass

    a = scalar()
    fg = FunctionGraph(outputs=[SomeOp()(a)])

    with pytest.raises(ValueError, match="`nodes` and `thunks`.*"):
        SomeVM(fg, fg.apply_nodes, [], [])


def test_Loop_exception():
    a = scalar()
    fg = FunctionGraph(outputs=[SomeOp()(a)])

    # Create valid(ish) `VM` arguments
    nodes = fg.toposort()
    input_storage, output_storage, storage_map = map_storage(
        fg, nodes, None, None, None
    )

    compute_map = {}
    for k in storage_map:
        compute_map[k] = [k.owner is None]

    thunks = [node.op.make_thunk(node, storage_map, compute_map, []) for node in nodes]

    with pytest.raises(ValueError, match="`nodes`, `thunks` and `post_thunk_clear`.*"):
        Loop(
            fg,
            fg.apply_nodes,
            thunks,
            [],
            storage_map,
            input_storage,
            output_storage,
            {},
            [],
        )


def test_Loop_updates():
    a = scalar("a")
    a_plus_1 = a + 1
    fg = FunctionGraph(outputs=[a, a_plus_1], clone=False)

    nodes = fg.toposort()
    input_storage, output_storage, storage_map = map_storage(
        fg, nodes, None, None, None
    )

    compute_map = {}
    for k in storage_map:
        compute_map[k] = [k.owner is None]

    thunks = [node.op.make_thunk(node, storage_map, compute_map, []) for node in nodes]

    assert a in storage_map

    update_vars = {a: a_plus_1}

    loop_vm = Loop(
        fg,
        fg.apply_nodes,
        thunks,
        [],
        storage_map,
        input_storage,
        output_storage,
        update_vars,
    )

    storage_map[a][0] = np.array(1.0, dtype=config.floatX)

    res = loop_vm()

    assert res == [np.array(1.0), np.array(2.0)]
    assert storage_map[a][0] == np.array(2.0)


def test_Stack_updates():
    a = scalar("a")
    a_plus_1 = a + 1
    fg = FunctionGraph(outputs=[a, a_plus_1], clone=False)

    nodes = fg.toposort()
    input_storage, output_storage, storage_map = map_storage(
        fg, nodes, None, None, None
    )

    compute_map = {}
    for k in storage_map:
        compute_map[k] = [k.owner is None]

    thunks = [node.op.make_thunk(node, storage_map, compute_map, []) for node in nodes]

    assert a in storage_map

    update_vars = {a: a_plus_1}

    stack_vm = Stack(
        fg,
        fg.apply_nodes,
        thunks,
        [],
        storage_map,
        input_storage,
        output_storage,
        update_vars,
        compute_map,
        False,
    )

    storage_map[a][0] = np.array(1.0, dtype=config.floatX)

    res = stack_vm()

    assert res == [np.array(1.0), np.array(2.0)]
    assert storage_map[a][0] == np.array(2.0)

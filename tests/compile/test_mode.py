import copy

import pytest

from aesara.compile.function import function
from aesara.compile.mode import (
    AddFeatureOptimizer,
    Mode,
    get_default_mode,
    get_target_language,
)
from aesara.configdefaults import config
from aesara.graph.features import NoOutputFromInplace
from aesara.graph.rewriting.db import RewriteDatabaseQuery, SequenceDB
from aesara.link.basic import LocalLinker
from aesara.tensor.math import dot, tanh
from aesara.tensor.type import matrix, vector


def test_Mode_basic():
    db = SequenceDB()
    mode = Mode(linker="py", optimizer=RewriteDatabaseQuery(include=None), db=db)

    assert mode.optdb is db

    assert str(mode).startswith("Mode(linker=py, optimizer=RewriteDatabaseQuery")


def test_NoOutputFromInplace():
    x = matrix()
    y = matrix()
    a = dot(x, y)
    b = tanh(a)
    c = tanh(dot(2 * x, y))

    # Ensure that the elemwise op that produces the output is inplace when
    # using a mode that does not include the optimization
    fct_no_opt = function([x, y], [b, c], mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert op.destroy_map and 0 in op.destroy_map
    op = fct_no_opt.maker.fgraph.outputs[1].owner.op
    assert op.destroy_map and 0 in op.destroy_map

    # Ensure that the elemwise op that produces the output is not inplace when
    # using a mode that includes the optimization
    opt = AddFeatureOptimizer(NoOutputFromInplace([1]))
    mode_opt = Mode(linker="py", optimizer="fast_run").register((opt, 49.9))

    fct_opt = function([x, y], [b, c], mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert op.destroy_map and 0 in op.destroy_map
    op = fct_opt.maker.fgraph.outputs[1].owner.op
    assert not op.destroy_map or 0 not in op.destroy_map


def test_including():
    mode = Mode(optimizer="merge")
    assert set(mode._optimizer.include) == {"merge"}

    new_mode = mode.including("fast_compile")
    assert set(new_mode._optimizer.include) == {"merge", "fast_compile"}


class TestBunchOfModes:
    def test_modes(self):
        # this is a quick test after the LazyLinker branch merge
        # to check that all the current modes can still be used.
        linker_classes_involved = []

        predef_modes = ["FAST_COMPILE", "FAST_RUN", "DEBUG_MODE"]

        # Linkers to use with regular Mode
        if config.cxx:
            linkers = ["py", "c|py", "c|py_nogc", "vm", "vm_nogc", "cvm", "cvm_nogc"]
        else:
            linkers = ["py", "c|py", "c|py_nogc", "vm", "vm_nogc"]
        modes = predef_modes + [Mode(linker, "fast_run") for linker in linkers]

        for mode in modes:
            x = matrix()
            y = vector()
            f = function([x, y], x + y, mode=mode)
            # test that it runs something
            f([[1, 2], [3, 4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)
            # print 'MODE:', mode, f.maker.mode.linker, 'stop'

        # regression check:
        # there should be
        # - `VMLinker`
        # - OpWiseCLinker (FAST_RUN)
        # - PerformLinker (FAST_COMPILE)
        # - DebugMode's Linker  (DEBUG_MODE)
        assert 4 == len(set(linker_classes_involved))


class TestOldModesProblem:
    def test_modes(self):
        # Then, build a mode with the same linker, and a modified optimizer
        default_mode = get_default_mode()
        modified_mode = default_mode.including("specialize")

        # The following line used to fail, with Python 2.4, in July 2012,
        # because an fgraph was associated to the default linker
        copy.deepcopy(modified_mode)

        # More straightforward test
        linker = get_default_mode().linker
        assert not hasattr(linker, "fgraph") or linker.fgraph is None


def test_get_target_language():
    with config.change_flags(mode=Mode(linker="py")):
        res = get_target_language()
        assert res == ("py",)

    res = get_target_language(Mode(linker="py"))
    assert res == ("py",)

    res = get_target_language(Mode(linker="c"))
    assert res == ("c",)

    res = get_target_language(Mode(linker="c|py"))
    assert res == ("c", "py")

    res = get_target_language(Mode(linker="vm"))
    assert res == ("c", "py")

    with config.change_flags(cxx=""):
        res = get_target_language(Mode(linker="vm"))
        assert res == ("py",)

    res = get_target_language(Mode(linker="jax"))
    assert res == ("jax",)

    res = get_target_language(Mode(linker="numba"))
    assert res == ("numba",)

    class MyLinker(LocalLinker):
        pass

    test_mode = Mode(linker=MyLinker())
    with pytest.raises(Exception):
        get_target_language(test_mode)

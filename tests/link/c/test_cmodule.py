"""
We don't have real tests for the cache, but it would be great to make them!

But this one tests a current behavior that isn't good: the c_code isn't
deterministic based on the input type and the op.
"""
import logging
import multiprocessing
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.compile.function import function
from aesara.compile.ops import DeepCopyOp
from aesara.configdefaults import config
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.link.c.basic import CLinker
from aesara.link.c.cmodule import GCC_compiler, ModuleCache, default_blas_ldflags
from aesara.link.c.exceptions import CompileError
from aesara.link.c.op import COp
from aesara.tensor.type import dvectors, vector


class MyOp(DeepCopyOp):
    nb_called = 0

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inames, onames, sub):
        MyOp.nb_called += 1
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]
        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            rand = np.random.random()
            return f'printf("{rand}\\n");{code % locals()}'
        # Else, no C code
        return super(DeepCopyOp, self).c_code(node, name, inames, onames, sub)


class MyAdd(COp):
    __props__ = ()

    def make_node(self, *inputs):
        outputs = [vector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, out_):
        (out,) = out_
        out[0] = inputs[0][0] + 1

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        return f"{z} = {x} + 1;"


class MyAddVersioned(MyAdd):
    def c_code_cache_version(self):
        return (1,)


def test_compiler_error():
    with pytest.raises(CompileError), tempfile.TemporaryDirectory() as dir_name:
        GCC_compiler.compile_str("module_name", "blah", location=dir_name)


def test_inter_process_cache():
    """
    TODO FIXME: This explanation is very poorly written.

    When a `COp` with `COp.c_code`, but no version. If we have two `Apply`
    nodes in a graph with distinct inputs variable, but the input variables
    have the same `Type`, do we reuse the same module? Even if they would
    generate different `COp.c_code`?  Currently this test show that we generate
    the `COp.c_code` only once.

    This is to know if the `COp.c_code` can add information specific to the
    ``node.inputs[*].owner`` like the name of the variable.

    """

    x, y = dvectors("xy")
    f = function([x, y], [MyOp()(x), MyOp()(y)])
    f(np.arange(60), np.arange(60))
    if config.mode == "FAST_COMPILE" or config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1

    # What if we compile a new function with new variables?
    x, y = dvectors("xy")
    f = function([x, y], [MyOp()(x), MyOp()(y)])
    f(np.arange(60), np.arange(60))
    if config.mode == "FAST_COMPILE" or config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1


@pytest.mark.filterwarnings("error")
def test_cache_versioning():
    """Make sure `ModuleCache._add_to_cache` is working."""

    my_add = MyAdd()
    with pytest.warns(match=".*specifies no C code cache version.*"):
        assert my_add.c_code_cache_version() == ()

    my_add_ver = MyAddVersioned()
    assert my_add_ver.c_code_cache_version() == (1,)

    assert len(MyOp.__props__) == 0
    assert len(MyAddVersioned.__props__) == 0

    x = vector("x")

    z = my_add(x)
    z_v = my_add_ver(x)

    with tempfile.TemporaryDirectory() as dir_name:
        cache = ModuleCache(dir_name)

        lnk = CLinker().accept(FunctionGraph(outputs=[z]))
        with pytest.warns(match=".*specifies no C code cache version.*"):
            key = lnk.cmodule_key()
        assert key[0] == ()

        with pytest.warns(match=".*c_code_cache_version.*"):
            cache.module_from_key(key, lnk)

        lnk_v = CLinker().accept(FunctionGraph(outputs=[z_v]))
        key_v = lnk_v.cmodule_key()
        assert len(key_v[0]) > 0

        assert key_v not in cache.entry_from_key

        stats_before = cache.stats[2]
        cache.module_from_key(key_v, lnk_v)
        assert stats_before < cache.stats[2]


def test_flag_detection():
    """
    TODO FIXME: This is a very poor test.

    Check that the code detecting blas flags does not raise any exception.
    It used to happen on Python 3 because of improper string handling,
    but was not detected because that path is not usually taken,
    so we test it here directly.
    """
    res = GCC_compiler.try_flags(["-lblas"])
    assert isinstance(res, bool)


@patch("aesara.link.c.cmodule.try_blas_flag", return_value=None)
@patch("aesara.link.c.cmodule.sys")
def test_default_blas_ldflags(sys_mock, try_blas_flag_mock, caplog):
    sys_mock.version = "3.8.0 | packaged by conda-forge | (default, Nov 22 2019, 19:11:38) \n[GCC 7.3.0]"

    with patch.dict("sys.modules", {"mkl": None}):
        with caplog.at_level(logging.WARNING):
            default_blas_ldflags()

    assert caplog.text == ""


@patch(
    "os.listdir", return_value=["mkl_core.1.dll", "mkl_rt.1.0.dll", "mkl_rt.1.1.lib"]
)
@patch("sys.platform", "win32")
def test_patch_ldflags(listdir_mock):
    mkl_path = "some_path"
    flag_list = ["-lm", "-lopenblas", f"-L {mkl_path}", "-l mkl_core", "-lmkl_rt"]
    assert GCC_compiler.patch_ldflags(flag_list) == [
        "-lm",
        "-lopenblas",
        f"-L {mkl_path}",
        '"' + os.path.join(mkl_path, "mkl_core.1.dll") + '"',
        '"' + os.path.join(mkl_path, "mkl_rt.1.0.dll") + '"',
    ]


@patch(
    "os.listdir",
    return_value=[
        "libopenblas.so",
        "libm.a",
        "mkl_core.1.dll",
        "mkl_rt.1.0.dll",
        "mkl_rt.1.1.dll",
    ],
)
@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_linking_patch(listdir_mock, platform):
    libs = ["openblas", "m", "mkl_core", "mkl_rt"]
    lib_dirs = ['"mock_dir"']
    with patch("sys.platform", platform):
        if platform == "win32":
            assert GCC_compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                '"' + os.path.join(lib_dirs[0].strip('"'), "mkl_core.1.dll") + '"',
                '"' + os.path.join(lib_dirs[0].strip('"'), "mkl_rt.1.1.dll") + '"',
            ]
        else:
            GCC_compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                "-lmkl_core",
                "-lmkl_rt",
            ]


@pytest.mark.skipif(sys.platform == "darwin", reason="Fails on MacOS")
def test_cache_race_condition():
    with tempfile.TemporaryDirectory() as dir_name:

        @config.change_flags(on_opt_error="raise", on_shape_error="raise")
        def f_build(factor):
            # Some of the caching issues arise during constant folding within the
            # optimization passes, so we need these config changes to prevent the
            # exceptions from being caught
            a = at.vector()
            f = aesara.function([a], factor * a)
            return f(np.array([1], dtype=config.floatX))

        ctx = multiprocessing.get_context()
        compiledir_prop = aesara.config._config_var_dict["compiledir"]

        # The module cache must (initially) be `None` for all processes so that
        # `ModuleCache.refresh` is called
        with patch.object(compiledir_prop, "val", dir_name, create=True), patch.object(
            aesara.link.c.cmodule, "_module_cache", None
        ):
            assert aesara.config.compiledir == dir_name

            num_procs = 30
            rng = np.random.default_rng(209)

            for i in range(10):
                # A random, constant input to prevent caching between runs
                factor = rng.random()
                procs = [
                    ctx.Process(target=f_build, args=(factor,))
                    for i in range(num_procs)
                ]
                for proc in procs:
                    proc.start()
                for proc in procs:
                    proc.join()

                assert not any(
                    exit_code != 0 for exit_code in [proc.exitcode for proc in procs]
                )


@patch("sys.platform", "darwin")
def test_osx_narrowing_compile_args():
    no_narrowing_flag = "-Wno-c++11-narrowing"
    assert no_narrowing_flag in GCC_compiler.compile_args()

    cxxflags = f"{aesara.config.gcc__cxxflags} {no_narrowing_flag}"
    with aesara.config.change_flags(gcc__cxxflags=cxxflags):
        print(cxxflags)
        res = GCC_compiler.compile_args()
        print(res)
        flag_idx = res.index(no_narrowing_flag)
        # Make sure it's not in there twice
        with pytest.raises(ValueError):
            res.index(no_narrowing_flag, flag_idx + 1)

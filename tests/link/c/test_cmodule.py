"""
We don't have real tests for the cache, but it would be great to make them!

But this one tests a current behavior that isn't good: the c_code isn't
deterministic based on the input type and the op.
"""
import logging
import multiprocessing
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

import aesara
import aesara.tensor as at
from aesara.compile.function import function
from aesara.compile.ops import DeepCopyOp
from aesara.configdefaults import config
from aesara.link.c.cmodule import GCC_Compiler, default_blas_ldflags
from aesara.link.c.exceptions import CompileError
from aesara.tensor.type import dvectors


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


def test_compiler_error():
    with pytest.raises(CompileError), tempfile.TemporaryDirectory() as dir_name:
        GCC_Compiler.compile_str("module_name", "blah", location=dir_name)


def test_inter_process_cache():
    # When an op with c_code, but no version. If we have 2 apply node
    # in the graph with different inputs variable(so they don't get
    # merged) but the inputs variable have the same type, do we reuse
    # the same module? Even if they would generate different c_code?
    # Currently this test show that we generate the c_code only once.
    #
    # This is to know if the c_code can add information specific to the
    # node.inputs[*].owner like the name of the variable.

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


def test_flag_detection():
    # Check that the code detecting blas flags does not raise any exception.
    # It used to happen on python 3 because of improper string handling,
    # but was not detected because that path is not usually taken,
    # so we test it here directly.
    GCC_Compiler.try_flags(["-lblas"])


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
    assert GCC_Compiler.patch_ldflags(flag_list) == [
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
            assert GCC_Compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                '"' + os.path.join(lib_dirs[0].strip('"'), "mkl_core.1.dll") + '"',
                '"' + os.path.join(lib_dirs[0].strip('"'), "mkl_rt.1.1.dll") + '"',
            ]
        else:
            GCC_Compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                "-lmkl_core",
                "-lmkl_rt",
            ]


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

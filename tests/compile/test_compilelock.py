import os
import shutil
import threading

import numpy as np

import aesara
import aesara.compile.compiledir
import aesara.link.c.cmodule
import aesara.tensor as at


def test_thread_safety():
    shutil.rmtree(aesara.config.compiledir)
    os.mkdir(aesara.config.compiledir)

    np.random.seed(32)

    def run_aesara(i, out):
        s = at.constant(np.random.randint(10, 50))
        a = at.random.normal(size=(s, s))
        fn = aesara.function([], a)
        for _ in range(10):
            fn()
        out[i] = True

    T = 32
    out = [None] * T

    threads = [threading.Thread(target=run_aesara, args=(i, out)) for i in range(T)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(out)


def test_compile_is_cached_for_multiple_threads(mocker):
    shutil.rmtree(aesara.config.compiledir)
    os.mkdir(aesara.config.compiledir)
    spy = mocker.spy(aesara.link.c.cmodule, "std_include_dirs")

    def run_aesara_compile(i, out):
        a = at.ones((3, 3))
        b = at.vector()
        o = b + a
        fg = aesara.link.basic.FunctionGraph([b], [o])
        lk = aesara.link.c.basic.CLinker().accept(fg)
        lk.compile_cmodule(lk.cmodule_key())
        out[i] = True

    T = 32
    out = [None] * T
    # running threads will always give the same graph. There should be only one compilation step
    threads = [
        threading.Thread(target=run_aesara_compile, args=(i, out)) for i in range(T)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert spy.call_count == 1
    assert all(out)


def test_compile_is_not_cached_for_no_key_modules(mocker):
    shutil.rmtree(aesara.config.compiledir)
    os.mkdir(aesara.config.compiledir)
    spy = mocker.spy(aesara.link.c.cmodule, "std_include_dirs")

    def run_aesara_compile(i, out):
        a = at.ones((3, 3))
        b = at.vector()
        o = b + a
        fg = aesara.link.basic.FunctionGraph([b], [o])
        lk = aesara.link.c.basic.CLinker().accept(fg)
        lk.compile_cmodule()
        out[i] = True

    T = 32
    out = [None] * T
    # running threads will always give the same graph. There should be only one compilation step
    threads = [
        threading.Thread(target=run_aesara_compile, args=(i, out)) for i in range(T)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert spy.call_count == T
    assert all(out)

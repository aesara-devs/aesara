import os
import shutil
import threading

import numpy as np

import aesara
import aesara.compile.compiledir
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

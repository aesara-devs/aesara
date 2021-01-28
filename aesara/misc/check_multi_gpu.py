#! /usr/bin/env python
"""
This file compare the runtime of two independent dot products on one
and two GPU to measure the speedup.

This should be 2x if the GPUs are equivalent.
"""

import threading
import time

import numpy as np

import aesara
from aesara.gpuarray import init_dev
from aesara.gpuarray.blas import gpu_dot22


def main(dev1, dev2):
    init_dev(dev1, "ctx1")
    init_dev(dev2, "ctx2")

    size = 1024 * 16
    data = np.random.randn(size, size).astype("float32")
    val1a = aesara.shared(data, target="ctx1")
    val1b = aesara.shared(data, target="ctx1")
    val1c = aesara.shared(data, target="ctx1")
    val1d = aesara.shared(data, target="ctx1")

    val2a = aesara.shared(data, target="ctx2")
    val2b = aesara.shared(data, target="ctx2")

    f1 = aesara.function([], [gpu_dot22(val1a, val1b), gpu_dot22(val1c, val1d)])
    f2 = aesara.function([], [gpu_dot22(val1a, val1b), gpu_dot22(val2a, val2b)])
    f3 = aesara.function([], [gpu_dot22(val1a, val1b)])
    f4 = aesara.function([], [gpu_dot22(val2a, val2b)])
    f5 = aesara.function([], [gpu_dot22(val1a, val1b)[0, 0].transfer("cpu")])
    f6 = aesara.function([], [gpu_dot22(val2a, val2b)[0, 0].transfer("cpu")])

    # pre-execute to load code to GPU.
    r = f1.fn()
    r[0].sync(), r[1].sync()
    r = f2.fn()
    r[0].sync(), r[1].sync()
    r = f3.fn()
    r[0].sync()
    r = f4.fn()
    r[0].sync()
    r = f5.fn()
    r = f6.fn()
    r = None

    t = time.time()
    r = f1.fn()
    r[0].sync(), r[1].sync()
    t2 = time.time()
    r = None

    print(f"one ctx async {t2 - t:f}")

    t = time.time()
    r = f2.fn()
    r[0].sync(), r[1].sync()
    t2 = time.time()
    r = None

    print(f"two ctx async {t2 - t:f}")

    t = time.time()
    r = f3.fn()
    r2 = f4.fn()
    r[0].sync()
    r2[0].sync()
    t2 = time.time()
    r = None

    print(f"two ctx, 2 fct async {t2 - t:f}")

    t = time.time()
    r = f5.fn()
    r2 = f6.fn()
    t2 = time.time()
    r = None
    print(f"two ctx, 2 fct with transfer {t2 - t:f}")

    # Multi-thread version
    class myThread(threading.Thread):
        def __init__(self, name, f, sync):
            threading.Thread.__init__(self)
            self.f = f
            self.name = name
            self.sync = sync

        def run(self):
            # print "Starting " + self.name
            # r = self.f.fn(n_calls=10)
            r = self.f()
            # print "End " + self.name
            if self.sync:
                r[0].sync()
            self.r = r
            # print "Exiting " + self.name

    thread1 = myThread("Thread-3", f3, True)
    thread2 = myThread("Thread-4", f4, True)
    t = time.time()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    t2 = time.time()

    print(f"two ctx, 2 fct async, 2 threads {t2 - t:f}")

    thread1 = myThread("Thread-5", f5, False)
    thread2 = myThread("Thread-6", f6, False)
    t = time.time()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    t2 = time.time()

    print(f"two ctx, 2 fct with transfer, 2 threads {t2 - t:f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise ValueError("This script require two device names.")
    main(sys.argv[1], sys.argv[2])

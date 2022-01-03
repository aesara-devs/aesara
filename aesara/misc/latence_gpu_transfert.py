import time

import numpy as np

import aesara


y = aesara.tensor.type.fvector()
x = aesara.shared(np.zeros(1, dtype="float32"))
f1 = aesara.function([y], updates={x: y})
f2 = aesara.function([], x.transfer("cpu"))
print(f1.maker.fgraph.toposort())
print(f2.maker.fgraph.toposort())
for i in (1, 10, 100, 1000, 10000, 100000, 1000000, 10000000):
    o = np.zeros(i, dtype="float32")
    t0 = time.time()
    f1(o)
    t1 = time.time()
    tf1 = t1 - t0
    t0 = time.time()
    f2()
    t1 = time.time()

    print("%8i %6.1f ns %7.1f ns" % (i, tf1 * 1e6, (t1 - t0) * 1e6))

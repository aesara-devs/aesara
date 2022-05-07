import numpy as np

import aesara

x, y, z = aesara.tensor.vectors("xyz")
f = aesara.function([x, y, z], [(x + y + z) * 2])
xv = np.random.random((10,)).astype(aesara.config.floatX)
yv = np.random.random((10,)).astype(aesara.config.floatX)
zv = np.random.random((10,)).astype(aesara.config.floatX)
f(xv, yv, zv)

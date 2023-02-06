import pickle

import numpy as np

from aesara.utils import HashableNDArray


def test_HashableNDArray():
    rng = np.random.default_rng(2392)

    x = rng.random(size=(3, 2))

    x_hnda_1 = x.view(HashableNDArray)
    x_hnda_2 = x.view(HashableNDArray)

    assert x_hnda_1 is not x_hnda_2
    assert x_hnda_1 == x_hnda_2
    assert hash(x_hnda_1) == hash(x_hnda_2)

    x_pkl = pickle.dumps(x_hnda_1)
    x_hnda_3 = pickle.loads(x_pkl)

    assert x_hnda_3 == x_hnda_1

    import weakref

    wd = weakref.WeakValueDictionary()
    wd[(1, 2)] = x_hnda_1
    wd[(2, 3)] = x_hnda_2
    assert wd[(1, 2)] is x_hnda_1
    del x_hnda_1

    assert (1, 2) not in wd

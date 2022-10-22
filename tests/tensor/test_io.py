import os

import numpy as np
import pytest

import aesara
from aesara import function
from aesara.graph.basic import Variable
from aesara.link.c.type import Generic
from aesara.tensor.io import load


class TestLoadTensor:
    def setup_method(self):
        self.data = np.arange(5, dtype=np.int32)
        self.filename = os.path.join(aesara.config.compiledir, "_test.npy")
        np.save(self.filename, self.data)

    def test_basic(self):
        path = Variable(Generic(), None)
        # Not specifying mmap_mode defaults to None, and the data is
        # copied into main memory
        x = load(path, "int32", (None,))
        y = x * 2
        fn = function([path], y)
        assert (fn(self.filename) == (self.data * 2)).all()

    def test_invalid_modes(self):
        # Modes 'r+', 'r', and 'w+' cannot work with Aesara, becausei
        # the output array may be modified inplace, and that should not
        # modify the original file.
        path = Variable(Generic(), None)
        for mmap_mode in ("r+", "r", "w+", "toto"):
            with pytest.raises(ValueError):
                load(path, "int32", (None,), mmap_mode)

    def test_copy_on_write(self):
        path = Variable(Generic(), None)
        # 'c' means "copy-on-write", which allow the array to be overwritten
        # by an inplace Op in the graph, without modifying the underlying
        # file.
        x = load(path, "int32", (None,), "c")
        # x ** 2 has been chosen because it will work inplace.
        y = (x**2).sum()
        fn = function([path], y)
        # Call fn() twice, to check that inplace ops do not cause trouble
        assert (fn(self.filename) == (self.data**2).sum()).all()
        assert (fn(self.filename) == (self.data**2).sum()).all()

    def test_memmap(self):
        path = Variable(Generic(), None)
        x = load(path, "int32", (None,), mmap_mode="c")
        fn = function([path], x)
        assert type(fn(self.filename)) == np.core.memmap

    def teardown_method(self):
        os.remove(os.path.join(aesara.config.compiledir, "_test.npy"))

import os
import shutil
from tempfile import mkdtemp

import numpy as np

import aesara
from aesara.misc.pkl_utils import StripPickler, dump, load
from aesara.sandbox.rng_mrg import MRG_RandomStream
from aesara.tensor.type import matrix


class TestDumpLoad:
    def setup_method(self):
        # Work in a temporary directory to avoid cluttering the repository
        self.origdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        # Get back to the original dir, and delete the temporary one
        os.chdir(self.origdir)
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test_dump_load_mrg(self):
        rng = MRG_RandomStream()

        with open("test", "wb") as f:
            dump(rng, f)

        with open("test", "rb") as f:
            rng = load(f)

        assert type(rng) == MRG_RandomStream

    def test_dump_zip_names(self):
        foo_1 = aesara.shared(0, name="foo")
        foo_2 = aesara.shared(1, name="foo")
        foo_3 = aesara.shared(2, name="foo")
        with open("model.zip", "wb") as f:
            dump((foo_1, foo_2, foo_3, np.array(3)), f)
        keys = list(np.load("model.zip").keys())
        assert keys == ["foo", "foo_2", "foo_3", "array_0", "pkl"]
        foo_3 = np.load("model.zip")["foo_3"]
        assert foo_3 == np.array(2)
        with open("model.zip", "rb") as f:
            foo_1, foo_2, foo_3, array = load(f)
        assert array == np.array(3)


class TestStripPickler:
    def setup_method(self):
        # Work in a temporary directory to avoid cluttering the repository
        self.origdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        # Get back to the original dir, and delete the temporary one
        os.chdir(self.origdir)
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test_basic(self):
        with open("test.pkl", "wb") as f:
            m = matrix()
            dest_pkl = "my_test.pkl"
            with open(dest_pkl, "wb") as f:
                strip_pickler = StripPickler(f, protocol=-1)
                strip_pickler.dump(m)

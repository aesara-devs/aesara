"""
Some pickle test when pygpu isn't there. The test when pygpu is
available are in test_type.py.

This is needed as we skip all the test file when pygpu isn't there in
regular test file.
"""

import os
import sys
from pickle import Unpickler

import numpy as np
import pytest

from aesara.configdefaults import config
from aesara.gpuarray.type import ContextNotDefined


try:
    import pygpu  # noqa: F401

    have_pygpu = True
except ImportError:
    have_pygpu = False


@pytest.mark.skip(reason="These tests relied on saved/versioned pickled files.")
@pytest.mark.skipif(have_pygpu, reason="pygpu active")
def test_unpickle_gpuarray_as_numpy_ndarray_flag1():
    oldflag = config.experimental__unpickle_gpu_on_cpu
    config.experimental__unpickle_gpu_on_cpu = False

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = "GpuArray.pkl"

        with open(os.path.join(testfile_dir, fname), "rb") as fp:
            u = Unpickler(fp, encoding="latin1")
            with pytest.raises((ImportError, ContextNotDefined)):
                u.load()
    finally:
        config.experimental__unpickle_gpu_on_cpu = oldflag


@pytest.mark.skip(reason="These tests relied on saved/versioned pickled files.")
def test_unpickle_gpuarray_as_numpy_ndarray_flag2():
    oldflag = config.experimental__unpickle_gpu_on_cpu
    config.experimental__unpickle_gpu_on_cpu = True

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = "GpuArray.pkl"

        with open(os.path.join(testfile_dir, fname), "rb") as fp:
            u = Unpickler(fp, encoding="latin1")
            try:
                mat = u.load()
            except ImportError:
                # Windows sometimes fail with nonsensical errors like:
                #   ImportError: No module named type
                #   ImportError: No module named copy_reg
                # when "type" and "copy_reg" are builtin modules.
                if sys.platform == "win32":
                    exc_type, exc_value, exc_trace = sys.exc_info()
                    raise
                raise

        assert isinstance(mat, np.ndarray)
        assert mat[0] == -42.0

    finally:
        config.experimental__unpickle_gpu_on_cpu = oldflag

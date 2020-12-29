import multiprocessing
import os
import sys
import tempfile

import filelock
import pytest

from theano.compile.compilelock import force_unlock, local_mem, lock_ctx


def test_compilelock_errors():
    with tempfile.TemporaryDirectory() as dir:
        with pytest.raises(ValueError):
            with lock_ctx(dir, timeout=0):
                pass
        with pytest.raises(ValueError):
            with lock_ctx(dir, timeout=-2):
                pass


def test_compilelock_force_unlock():
    with tempfile.TemporaryDirectory() as dir_name:
        with lock_ctx(dir_name):
            dir_key = f"{dir_name}-{os.getpid()}"

            assert dir_key in local_mem._locks
            assert local_mem._locks[dir_key]

            force_unlock(dir_name)

            assert dir_key not in local_mem._locks

            # A sub-process forcing unlock...
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(target=force_unlock, args=(dir_name,))
            p.start()
            p.join()

            assert dir_key not in local_mem._locks


def check_is_locked(dir_name, q):
    try:
        with lock_ctx(dir_name, timeout=0.1):
            q.put("unlocked")
    except filelock.Timeout:
        q.put("locked")


def get_subprocess_lock_state(ctx, dir_name):
    q = ctx.Queue()
    p = ctx.Process(target=check_is_locked, args=(dir_name, q))
    p.start()
    result = q.get()
    p.join()
    return result


def run_locking_test(ctx):

    with tempfile.TemporaryDirectory() as dir_name:
        assert get_subprocess_lock_state(ctx, dir_name) == "unlocked"

        # create a lock on the test directory
        with lock_ctx(dir_name):
            dir_key = f"{dir_name}-{os.getpid()}"
            assert dir_key in local_mem._locks
            assert local_mem._locks[dir_key]

            assert get_subprocess_lock_state(ctx, dir_name) == "locked"

            with lock_ctx(dir_name, timeout=0.1):
                assert get_subprocess_lock_state(ctx, dir_name) == "locked"

            assert get_subprocess_lock_state(ctx, dir_name) == "locked"

        assert get_subprocess_lock_state(ctx, dir_name) == "unlocked"


@pytest.mark.skipif(sys.platform != "linux", reason="Fork is only available on linux")
def test_locking_multiprocess_fork():
    ctx = multiprocessing.get_context("fork")
    run_locking_test(ctx)


def test_locking_multiprocess_spawn():
    ctx = multiprocessing.get_context("spawn")
    run_locking_test(ctx)

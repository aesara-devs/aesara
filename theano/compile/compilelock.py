"""
Locking mechanism to ensure no two compilations occur simultaneously
in the same compilation directory (which can cause crashes).
"""
import os
import threading
import typing
from contextlib import contextmanager

import filelock

from theano.configdefaults import config


__all__ = [
    "force_unlock",
    "lock_ctx",
]


class ThreadFileLocks(threading.local):
    def __init__(self):
        self._locks = {}


local_mem = ThreadFileLocks()


def force_unlock(lock_dir: os.PathLike):
    """Forces the release of the lock on a specific directory.

    Parameters
    ----------
    lock_dir : os.PathLike
        Path to a directory that was locked with `lock_ctx`.
    """

    fl = filelock.FileLock(os.path.join(lock_dir, ".lock"))
    fl.release(force=True)

    dir_key = f"{lock_dir}-{os.getpid()}"

    if dir_key in local_mem._locks:
        del local_mem._locks[dir_key]


@contextmanager
def lock_ctx(lock_dir: os.PathLike = None, *, timeout: typing.Optional[float] = -1):
    """Context manager that wraps around FileLock and SoftFileLock from filelock package.

    Parameters
    ----------
    lock_dir : str
        A directory for which to acquire the lock.
        Defaults to the config.compiledir.
    timeout : float
        Timeout in seconds for waiting in lock acquisition.
        Defaults to config.compile__timeout.
    """
    if lock_dir is None:
        lock_dir = config.compiledir
    if timeout == -1:
        timeout = config.compile__timeout
    elif not (timeout is None or timeout > 0):
        raise ValueError(f"Timeout parameter must be None or positive. Got {timeout}.")

    # locks are kept in a dictionary to account for changing compiledirs
    dir_key = f"{lock_dir}-{os.getpid()}"

    if dir_key not in local_mem._locks:
        local_mem._locks[dir_key] = True
        fl = filelock.FileLock(os.path.join(lock_dir, ".lock"))
        fl.acquire(timeout=timeout)
        try:
            yield
        finally:
            if fl.is_locked:
                fl.release()
            if dir_key in local_mem._locks:
                del local_mem._locks[dir_key]
    else:
        yield

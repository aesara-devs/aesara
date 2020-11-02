"""Utility functions for Theano."""


from collections import OrderedDict
from collections.abc import Callable


__all__ = [
    "cmp",
    "decode",
    "decode_with",
    "decode_iter",
    "get_unbound_function",
    "maybe_add_to_os_environ_pathlist",
    "DefaultOrderedDict",
]


# In python 3.x, when an exception is reraised it saves original
# exception in its args, therefore in order to find the actual
# message, we need to unpack arguments recursively.
def exc_message(e):
    msg = e.args[0]
    if isinstance(msg, Exception):
        return exc_message(msg)
    return msg


def cmp(x, y):
    """Return -1 if x < y, 0 if x == y, 1 if x > y."""
    return (x > y) - (x < y)


def get_unbound_function(unbound):
    # Op.make_thunk isn't bound, so don't have a __func__ attr.
    # But bound method, have a __func__ method that point to the
    # not bound method. That is what we want.
    if hasattr(unbound, "__func__"):
        return unbound.__func__
    return unbound


def decode(x):
    return x.decode()


def decode_iter(itr):
    for x in itr:
        yield x.decode()


def decode_with(x, encoding):
    return x.decode(encoding)


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        if default_factory is not None and not isinstance(default_factory, Callable):
            raise TypeError("first argument must be callable")
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, list(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)


def maybe_add_to_os_environ_pathlist(var, newpath):
    """Unfortunately, Conda offers to make itself the default Python
    and those who use it that way will probably not activate envs
    correctly meaning e.g. mingw-w64 g++ may not be on their PATH.

    This function ensures that, if `newpath` is an absolute path,
    and it is not already in os.environ[var] it gets added to the
    front.

    The reason we check first is because Windows environment vars
    are limited to 8191 characters and it is easy to hit that.

    `var` will typically be 'PATH'."""

    import os

    if os.path.isabs(newpath):
        try:
            oldpaths = os.environ[var].split(os.pathsep)
            if newpath not in oldpaths:
                newpaths = os.pathsep.join([newpath] + oldpaths)
                os.environ[var] = newpaths
        except Exception:
            pass

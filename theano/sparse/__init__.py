from warnings import warn


try:
    import scipy

    scipy_ver = [int(n) for n in scipy.__version__.split(".")[:2]]
    enable_sparse = bool(scipy_ver >= [0, 7])
    if not enable_sparse:
        warn(f"SciPy version is {scipy.__version__}.  We recommend a version >= 0.7.0")
except ImportError:
    enable_sparse = False
    warn("scipy can't be imported." " We disable the sparse matrix code.")

from theano.sparse.type import *


if enable_sparse:
    from theano.sparse import opt, sharedvar
    from theano.sparse.basic import *
    from theano.sparse.sharedvar import sparse_constructor as shared

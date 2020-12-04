from warnings import warn


try:
    import scipy

    enable_sparse = True
except ImportError:
    enable_sparse = False
    warn("SciPy can't be imported.  Sparse matrix support is disabled.")

from theano.sparse.type import *


if enable_sparse:
    from theano.sparse import opt, sharedvar
    from theano.sparse.basic import *
    from theano.sparse.sharedvar import sparse_constructor as shared

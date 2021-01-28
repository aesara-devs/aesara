from warnings import warn


try:
    import scipy

    enable_sparse = True
except ImportError:
    enable_sparse = False
    warn("SciPy can't be imported.  Sparse matrix support is disabled.")

from aesara.sparse.type import SparseType, _is_sparse


if enable_sparse:
    from aesara.sparse import opt, sharedvar
    from aesara.sparse.basic import *
    from aesara.sparse.sharedvar import sparse_constructor as shared

    def sparse_grad(var):
        """This function return a new variable whose gradient will be
        stored in a sparse format instead of dense.

        Currently only variable created by AdvancedSubtensor1 is supported.
        i.e. a_tensor_var[an_int_vector].

        .. versionadded:: 0.6rc4
        """
        from aesara.tensor.subtensor import AdvancedSubtensor1

        assert isinstance(var.owner.op, AdvancedSubtensor1)

        ret = var.owner.op.__class__(sparse_grad=True)(*var.owner.inputs)
        return ret

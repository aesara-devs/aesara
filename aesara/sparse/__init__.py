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
        from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1

        if var.owner is None or not isinstance(
            var.owner.op, (AdvancedSubtensor, AdvancedSubtensor1)
        ):
            raise TypeError(
                "Sparse gradient is only implemented for AdvancedSubtensor and AdvancedSubtensor1"
            )

        x = var.owner.inputs[0]
        indices = var.owner.inputs[1:]

        if len(indices) > 1:
            raise TypeError(
                "Sparse gradient is only implemented for single advanced indexing"
            )

        ret = AdvancedSubtensor1(sparse_grad=True)(x, indices[0])
        return ret

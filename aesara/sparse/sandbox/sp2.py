import numpy as np
import scipy.sparse

import aesara
from aesara import tensor as at
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.sparse.basic import (
    Remove0,
    SparseType,
    _is_sparse,
    as_sparse_variable,
    remove0,
)
from aesara.tensor.type import discrete_dtypes, float_dtypes


# Probability Ops are currently back in sandbox, because they do not respect
# Aesara's Op contract, as their behaviour is not reproducible: calling
# the perform() method twice with the same argument will yield different
# results.
# from aesara.sparse.basic import (
#    Multinomial, multinomial, Poisson, poisson,
#    Binomial, csr_fbinomial, csc_fbinomial, csr_dbinomial, csc_dbinomial)


# Alias to maintain compatibility
EliminateZeros = Remove0
eliminate_zeros = remove0


# Probability
class Poisson(Op):
    """Return a sparse having random values from a Poisson density
    with mean from the input.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Aesara's contract for Ops

    :param x: Sparse matrix.

    :return: A sparse matrix of random integers of a Poisson density
             with mean of `x` element wise.
    """

    __props__ = ()

    def make_node(self, x):
        x = as_sparse_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        assert x.format in ("csr", "csc")
        out[0] = x.copy()
        out[0].data = np.asarray(np.random.poisson(out[0].data), dtype=x.dtype)
        out[0].eliminate_zeros()

    def grad(self, inputs, outputs_gradients):
        comment = "No gradient exists for class Poisson in\
                   aesara/sparse/sandbox/sp2.py"
        return [
            aesara.gradient.grad_undefined(
                op=self, x_pos=0, x=inputs[0], comment=comment
            )
        ]

    def infer_shape(self, fgraph, node, ins_shapes):
        return ins_shapes


poisson = Poisson()


class Binomial(Op):
    """Return a sparse matrix having random values from a binomial
    density having number of experiment `n` and probability of success
    `p`.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Aesara's contract for Ops

    :param n: Tensor scalar representing the number of experiment.
    :param p: Tensor scalar representing the probability of success.
    :param shape: Tensor vector for the output shape.

    :return: A sparse matrix of integers representing the number
             of success.
    """

    __props__ = ("format", "dtype")

    def __init__(self, format, dtype):
        self.format = format
        self.dtype = dtype

    def make_node(self, n, p, shape):
        n = at.as_tensor_variable(n)
        p = at.as_tensor_variable(p)
        shape = at.as_tensor_variable(shape)

        assert n.dtype in discrete_dtypes
        assert p.dtype in float_dtypes
        assert shape.dtype in discrete_dtypes

        return Apply(
            self, [n, p, shape], [SparseType(dtype=self.dtype, format=self.format)()]
        )

    def perform(self, node, inputs, outputs):
        (n, p, shape) = inputs
        (out,) = outputs
        binomial = np.random.binomial(n, p, size=shape)
        csx_matrix = getattr(scipy.sparse, self.format + "_matrix")
        out[0] = csx_matrix(binomial, dtype=self.dtype)

    def connection_pattern(self, node):
        return [[True], [True], [False]]

    def grad(self, inputs, gout):
        (n, p, shape) = inputs
        (gz,) = gout
        comment_n = "No gradient exists for the number of samples in class\
                     Binomial of aesara/sparse/sandbox/sp2.py"
        comment_p = "No gradient exists for the prob of success in class\
                     Binomial of aesara/sparse/sandbox/sp2.py"
        return [
            aesara.gradient.grad_undefined(op=self, x_pos=0, x=n, comment=comment_n),
            aesara.gradient.grad_undefined(op=self, x_pos=1, x=p, comment=comment_p),
            aesara.gradient.disconnected_type(),
        ]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [(node.inputs[2][0], node.inputs[2][1])]


csr_fbinomial = Binomial("csr", "float32")
csc_fbinomial = Binomial("csc", "float32")
csr_dbinomial = Binomial("csr", "float64")
csc_dbinomial = Binomial("csc", "float64")


class Multinomial(Op):
    """Return a sparse matrix having random values from a multinomial
    density having number of experiment `n` and probability of success
    `p`.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Aesara's contract for Ops

    :param n: Tensor type vector or scalar representing the number of
              experiment for each row. If `n` is a scalar, it will be
              used for each row.
    :param p: Sparse matrix of probability where each row is a probability
              vector representing the probability of success. N.B. Each row
              must sum to one.

    :return: A sparse matrix of random integers from a multinomial density
             for each row.

    :note: It will works only if `p` have csr format.
    """

    __props__ = ()

    def make_node(self, n, p):
        n = at.as_tensor_variable(n)
        p = as_sparse_variable(p)
        assert p.format in ("csr", "csc")

        return Apply(self, [n, p], [p.type()])

    def perform(self, node, inputs, outputs):
        (n, p) = inputs
        (out,) = outputs
        assert _is_sparse(p)

        if p.format != "csr":
            raise NotImplementedError

        out[0] = p.copy()

        if n.ndim == 0:
            for i in range(p.shape[0]):
                k, l = p.indptr[i], p.indptr[i + 1]
                out[0].data[k:l] = np.random.multinomial(n, p.data[k:l])
        elif n.ndim == 1:
            if n.shape[0] != p.shape[0]:
                raise ValueError(
                    "The number of element of n must be "
                    "the same as the number of row of p."
                )
            for i in range(p.shape[0]):
                k, l = p.indptr[i], p.indptr[i + 1]
                out[0].data[k:l] = np.random.multinomial(n[i], p.data[k:l])

    def grad(self, inputs, outputs_gradients):
        comment_n = "No gradient exists for the number of samples in class\
                     Multinomial of aesara/sparse/sandbox/sp2.py"
        comment_p = "No gradient exists for the prob of success in class\
                     Multinomial of aesara/sparse/sandbox/sp2.py"
        return [
            aesara.gradient.grad_undefined(
                op=self, x_pos=0, x=inputs[0], comment=comment_n
            ),
            aesara.gradient.grad_undefined(
                op=self, x_pos=1, x=inputs[1], comment=comment_p
            ),
        ]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [ins_shapes[1]]


multinomial = Multinomial()

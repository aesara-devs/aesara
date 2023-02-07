
.. _libdoc_sparse:

=========================================
:mod:`sparse` -- Symbolic Sparse Matrices
=========================================

In the tutorial section, you can find a :ref:`sparse tutorial
<tutsparse>`.

The sparse submodule is not loaded when we import Aesara. You must
import ``aesara.sparse`` to enable it.

The sparse module provides the same functionality as the tensor
module. The difference lies under the covers because sparse matrices
do not store data in a contiguous array. The sparse module has
been used in:

- NLP: Dense linear transformations of sparse vectors.
- Audio: Filterbank in the Fourier domain.

Compressed Sparse Format
========================

This section tries to explain how information is stored for the two
sparse formats of SciPy supported by Aesara.

.. Changes to this section should also result in changes to tutorial/sparse.txt.

Aesara supports two *compressed sparse formats*: ``csc`` and ``csr``,
respectively based on columns and rows. They have both the same
attributes: ``data``, ``indices``, ``indptr`` and ``shape``.

  * The ``data`` attribute is a one-dimensional ``ndarray`` which
    contains all the non-zero elements of the sparse matrix.

  * The ``indices`` and ``indptr`` attributes are used to store the
    position of the data in the sparse matrix.

  * The ``shape`` attribute is exactly the same as the ``shape``
    attribute of a dense (i.e. generic) matrix. It can be explicitly
    specified at the creation of a sparse matrix if it cannot be
    inferred from the first three attributes.


CSC Matrix
----------

In the *Compressed Sparse Column* format, ``indices`` stands for
indexes inside the column vectors of the matrix and ``indptr`` tells
where the column starts in the ``data`` and in the ``indices``
attributes. ``indptr`` can be thought of as giving the slice which
must be applied to the other attribute in order to get each column of
the matrix. In other words, ``slice(indptr[i], indptr[i+1])``
corresponds to the slice needed to find the i-th column of the matrix
in the ``data`` and ``indices`` fields.

The following example builds a matrix and returns its columns. It
prints the i-th column, i.e. a list of indices in the column and their
corresponding value in the second list.

>>> import numpy as np
>>> import scipy.sparse as sp
>>> data = np.asarray([7, 8, 9])
>>> indices = np.asarray([0, 1, 2])
>>> indptr = np.asarray([0, 2, 3, 3])
>>> m = sp.csc_matrix((data, indices, indptr), shape=(3, 3))
>>> m.toarray()
array([[7, 0, 0],
       [8, 0, 0],
       [0, 9, 0]])
>>> i = 0
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([0, 1], dtype=int32), array([7, 8]))
>>> i = 1
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([2], dtype=int32), array([9]))
>>> i = 2
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([], dtype=int32), array([], dtype=int64))

CSR Matrix
----------

In the *Compressed Sparse Row* format, ``indices`` stands for indexes
inside the row vectors of the matrix and ``indptr`` tells where the
row starts in the ``data`` and in the ``indices``
attributes. ``indptr`` can be thought of as giving the slice which
must be applied to the other attribute in order to get each row of the
matrix. In other words, ``slice(indptr[i], indptr[i+1])`` corresponds
to the slice needed to find the i-th row of the matrix in the ``data``
and ``indices`` fields.

The following example builds a matrix and returns its rows. It prints
the i-th row, i.e. a list of indices in the row and their
corresponding value in the second list.

>>> import numpy as np
>>> import scipy.sparse as sp
>>> data = np.asarray([7, 8, 9])
>>> indices = np.asarray([0, 1, 2])
>>> indptr = np.asarray([0, 2, 3, 3])
>>> m = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
>>> m.toarray()
array([[7, 8, 0],
       [0, 0, 9],
       [0, 0, 0]])
>>> i = 0
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([0, 1], dtype=int32), array([7, 8]))
>>> i = 1
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([2], dtype=int32), array([9]))
>>> i = 2
>>> m.indices[m.indptr[i]:m.indptr[i+1]], m.data[m.indptr[i]:m.indptr[i+1]]
(array([], dtype=int32), array([], dtype=int64))

List of Implemented Operations
==============================

- Moving from and to sparse
    - :func:`dense_from_sparse <aesara.sparse.basic.dense_from_sparse>`.
      Both grads are implemented. Structured by default.
    - :func:`csr_from_dense <aesara.sparse.basic.csr_from_dense>`,
      :func:`csc_from_dense <aesara.sparse.basic.csc_from_dense>`.
      The grad implemented is structured.
    - Aesara SparseVariable objects have a method ``toarray()`` that is the same as
      :func:`dense_from_sparse <aesara.sparse.basic.dense_from_sparse>`.

- Construction of Sparses and their Properties
    - :class:`CSM <aesara.sparse.basic.CSM>` and ``CSC``, ``CSR`` to construct a matrix.
      The grad implemented is regular.
    - :func:`csm_properties <aesara.sparse.basic.csm_properties>`.
      to get the properties of a sparse matrix.
      The grad implemented is regular.
    - csm_indices(x), csm_indptr(x), csm_data(x) and csm_shape(x) or x.shape.
    - :func:`sp_ones_like <aesara.sparse.basic.sp_ones_like>`.
      The grad implemented is regular.
    - :func:`sp_zeros_like <aesara.sparse.basic.sp_zeros_like>`.
      The grad implemented is regular.
    - :func:`square_diagonal <aesara.sparse.basic.square_diagonal>`.
      The grad implemented is regular.
    - :func:`construct_sparse_from_list <aesara.sparse.basic.construct_sparse_from_list>`.
      The grad implemented is regular.

- Cast
    - :func:`cast <aesara.sparse.basic.cast>` with ``bcast``, ``wcast``, ``icast``, ``lcast``,
      ``fcast``, ``dcast``, ``ccast``, and ``zcast``.
      The grad implemented is regular.

- Transpose
    - :func:`transpose <aesara.sparse.basic.transpose>`.
      The grad implemented is regular.

- Basic Arithmetic
    - :func:`neg <aesara.sparse.basic.neg>`.
      The grad implemented is regular.
    - :func:`eq <aesara.sparse.basic.eq>`.
    - :func:`neq <aesara.sparse.basic.neq>`.
    - :func:`gt <aesara.sparse.basic.gt>`.
    - :func:`ge <aesara.sparse.basic.ge>`.
    - :func:`lt <aesara.sparse.basic.lt>`.
    - :func:`le <aesara.sparse.basic.le>`.
    - :func:`add <aesara.sparse.basic.add>`.
      The grad implemented is regular.
    - :func:`sub <aesara.sparse.basic.sub>`.
      The grad implemented is regular.
    - :func:`mul <aesara.sparse.basic.mul>`.
      The grad implemented is regular.
    - :func:`col_scale <aesara.sparse.basic.col_scale>` to multiply by a vector along the columns.
      The grad implemented is structured.
    - :func:`row_scale <aesara.sparse.basic.row_scale>` to multiply by a vector along the rows.
      The grad implemented is structured.

- Monoid (Element-wise operation with only one sparse input).
    `They all have a structured grad.`

    - ``structured_sigmoid``
    - ``structured_exp``
    - ``structured_log``
    - ``structured_pow``
    - ``structured_minimum``
    - ``structured_maximum``
    - ``structured_add``
    - ``sin``
    - ``arcsin``
    - ``tan``
    - ``arctan``
    - ``sinh``
    - ``arcsinh``
    - ``tanh``
    - ``arctanh``
    - ``rad2deg``
    - ``deg2rad``
    - ``rint``
    - ``ceil``
    - ``floor``
    - ``trunc``
    - ``sgn``
    - ``log1p``
    - ``expm1``
    - ``sqr``
    - ``sqrt``

- Dot Product
    - :func:`dot <aesara.sparse.basic.dot>`.

        - One of the inputs must be sparse, the other sparse or dense.
        - The grad implemented is regular.
        - No C code for perform and no C code for grad.
        - Returns a dense for perform and a dense for grad.
    - :func:`structured_dot <aesara.sparse.basic.structured_dot>`.

        - The first input is sparse, the second can be sparse or dense.
        - The grad implemented is structured.
        - C code for perform and grad.
        - It returns a sparse output if both inputs are sparse and
          dense one if one of the inputs is dense.
        - Returns a sparse grad for sparse inputs and dense grad for
          dense inputs.
    - :func:`true_dot <aesara.sparse.basic.true_dot>`.

        - The first input is sparse, the second can be sparse or dense.
        - The grad implemented is regular.
        - No C code for perform and no C code for grad.
        - Returns a Sparse.
        - The gradient returns a Sparse for sparse inputs and by
          default a dense for dense inputs. The parameter
          ``grad_preserves_dense`` can be set to False to return a
          sparse grad for dense inputs.
    - :func:`sampling_dot <aesara.sparse.basic.sampling_dot>`.

        - Both inputs must be dense.
        - The grad implemented is structured for ``p``.
        - Sample of the dot and sample of the gradient.
        - C code for perform but not for grad.
        - Returns sparse for perform and grad.
    - :func:`usmm <aesara.sparse.basic.usmm>`.

        - You *shouldn't* insert this op yourself!
           - There is a rewrite that transforms a
             :func:`dot <aesara.sparse.basic.dot>` to :class:`Usmm` when possible.

        - This :class:`Op` is the equivalent of gemm for sparse dot.
        - There is no grad implemented for this :class:`Op`.
        - One of the inputs must be sparse, the other sparse or dense.
        - Returns a dense from perform.

- Slice Operations
    - sparse_variable[N, N], returns a tensor scalar.
      There is no grad implemented for this operation.
    - sparse_variable[M:N, O:P], returns a sparse matrix
      There is no grad implemented for this operation.
    - Sparse variables don't support [M, N:O] and [M:N, O] as we don't
      support sparse vectors and returning a sparse matrix would break
      the numpy interface.  Use [M:M+1, N:O] and [M:N, O:O+1] instead.
    - :func:`diag <aesara.sparse.basic.diag>`.
      The grad implemented is regular.

- Concatenation
    - :func:`hstack <aesara.sparse.basic.hstack>`.
      The grad implemented is regular.
    - :func:`vstack <aesara.sparse.basic.vstack>`.
      The grad implemented is regular.

- Probability
    `There is no grad implemented for these operations.`

    - :class:`Poisson <aesara.sparse.basic.Poisson>` and ``poisson``
    - :class:`Binomial <aesara.sparse.basic.Binomial>` and ``csc_fbinomial``, ``csc_dbinomial``
      ``csr_fbinomial``, ``csr_dbinomial``
    - :class:`Multinomial <aesara.sparse.basic.Multinomial>` and ``multinomial``

- Internal Representation
    `They all have a regular grad implemented.`

    - :func:`ensure_sorted_indices <aesara.sparse.basic.ensure_sorted_indices>`.
    - :func:`remove0 <aesara.sparse.basic.remove0>`.
    - :func:`clean <aesara.sparse.basic.clean>` to resort indices and remove zeros

- To help testing
    - :func:`tests.sparse.test_basic.sparse_random_inputs`

===================================================================
:mod:`sparse` --  Sparse Op
===================================================================

.. module:: sparse
   :platform: Unix, Windows
   :synopsis: Sparse Op
.. moduleauthor:: LISA

.. automodule:: aesara.sparse.basic
    :members:

.. autofunction:: aesara.sparse.sparse_grad

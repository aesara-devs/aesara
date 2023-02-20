.. _other_ops:

==============================
Implementing some specific Ops
==============================

This page is a guide on the implementation of some specific types of Ops,
and points to some examples of such implementations.

For the random number generating Ops, it explains different possible
implementation strategies.


.. _scalar_ops:

Scalar/Elemwise/Reduction Ops
=============================

Implementing an Aesara scalar Op allows that scalar operation to be reused
by our elemwise operations on tensors. If the scalar operation has C code, the
elemwise implementation will automatically have C code too. This
will enable the fusion of elemwise operations using your new scalar
operation. It is similar for reduction operations.

Be careful about some possible problems in the definition of the
``grad`` method, and about dependencies that may not be available. In
particular, see the following fixes:
`Fix to grad() methods
<https://github.com/Theano/Theano/commit/002872ad97919b97eaf58e095044e3c3067668e4>`_
and `impl() methods related to SciPy
<https://github.com/Theano/Theano/commit/08d16c0aa6681fc53d8d0f40342551eb47ff536e>`_.

.. _sparse_ops:

Sparse Ops
==========

There are a few differences to keep in mind if you want to make an op
that uses :ref:`sparse <tutsparse>` inputs or outputs, rather than the
usual dense tensors. In particular, in the
``make_node()`` function, you have to call
``Aesara.sparse.as_sparse_variable(x)`` on sparse input variables,
instead of ``as_tensor_variable(x)``.

Another difference is that you need to use ``SparseVariable`` and
``SparseTensorType`` instead of ``TensorVariable`` and ``TensorType``.

Do not forget that we support only sparse matrices (so only 2 dimensions)
and (like in SciPy) they do not support broadcasting operations by default
(although a few Ops do it when called manually). Also, we support only two
formats for sparse type: ``csr`` and ``csc``. So in ``make_mode()``,
you can create output variables like this:

.. code-block:: python

    out_format = inputs[0].format  # or 'csr' or 'csc' if the output format is fixed
    SparseTensorType(dtype=inputs[0].dtype, format=out_format).make_variable()

See the sparse :class:`Aesara.sparse.basic.Cast` `Op` code for a good example of
a sparse `Op` with Python code.

.. note::

   From the definition of CSR and CSC formats, CSR column indices are
   not necessarily sorted. Likewise for CSC row indices. Use
   :class:`EnsureSortedIndices
   <Aesara.sparse.basic.EnsureSortedIndices>` if your code does not
   support it.

   Also, there can be explicit zeros in your inputs. Use
   :class:`Remove0 <Aesara.sparse.basic.Remove0>` or ``remove0`` to
   make sure they aren't present in your input if you don't support
   that.

   To remove explicit zeros and make sure indices are sorted, use
   :func:`clean <Aesara.sparse.basic.clean>`.

Sparse Gradient
---------------

There are 2 types of :ref:`gradients <tutsparse_gradient>` for sparse
operations: ``normal``
gradient and ``structured`` gradient. Please document what your op
implements in its docstring. It is important that the user knows it, and
it is not always easy to infer from the code. Also make clear which
inputs/outputs are sparse and which ones are dense.

Sparse C code
-------------

Aesara does not have a native C code interface for sparse matrices. The
reason is simple: we use the SciPy sparse matrix objects and they don't
have a C object. So we use a simple trick: a sparse matrix is made of
4 fields that are NumPy vector arrays: ``data``, ``indices``, ``indptr``
and ``shape``. So to make
an op with C code that has sparse variables as inputs, we actually make an op
that takes as input the needed fields of those sparse variables.

You can extract the 4 fields with
:func:`Aesara.sparse.basic.csm_properties`. You can use
:func:`Aesara.sparse.basic.csm_data`,
:func:`Aesara.sparse.basic.csm_indices`,
:func:`Aesara.sparse.basic.csm_indptr` and
:func:`Aesara.sparse.basic.csm_shape` to extract the individual
fields.

You can look at the `AddSD` sparse `Op` for an example with C code. It implements
the addition of a sparse matrix with a dense matrix.

Sparse Tests
------------

You can reuse the test system for tensor variables. To generate the
needed sparse variable and data, you can use
:func:`tests.sparse.test_basic.sparse_random_inputs`. It takes
many parameters, including parameters for the format (csr or csc), the shape, the
dtype, whether to have explicit 0 and whether to have unsorted indices.

.. _random_ops:

Random distribution
===================

We have 3 base random number generators. One that wraps NumPy's random
generator, one that implements MRG31k3p and one that wraps CURAND.

The recommended and 2nd faster is MRG. It works on the CPU and
has more implemented distributions.

The slowest is our wrapper on NumPy's random generator.

We explain and provide advice on 3 possibles implementations of new
distributions here:

1. Extend our wrapper around NumPy random functions.
   See this `PR <https://github.com/Theano/Theano/pull/1607>`_ as an example.

2. Extend MRG implementation by reusing existing Aesara Op. Look into
   the ``Aesara/sandbox/rng_mrg.py`` file and grep for all code about
   binomial(). This distribution uses the output of the uniform
   distribution and converts it to a binomial distribution with
   existing Aesara operations. The tests go in
   ``Aesara/sandbox/test_rng_mrg.py``

3. Extend MRG implementation with a new Op that takes a uniform sample as
   input. Look in the ``Aesara/sandbox/{rng_mrg,multinomial}.py`` file
   and its test in ``Aesara/sandbox/test_multinomal.py``. This is
   recommended when current Aesara ops aren't well suited to modify
   the uniform to the target distribution. This can happen in
   particular if there is a loop or complicated condition.

.. note::

    In all cases, you must reuse the same interface as NumPy for compatibility.


.. _openmp_ops:

OpenMP Ops
==========

To allow consistent interface of Ops that support OpenMP, we have some
helper code. Doing this also allows to enable/disable OpenMP globally
or per op for fine-grained control.

Your Op needs to inherit from ``aesara.link.c.op.OpenMPOp``. If it overrides
the ``__init__()`` method, it must have an ``openmp=None`` parameter
and must call ``super(MyOpClass, self).__init__(openmp=openmp)``.

The ``OpenMPOp`` class also implements ``c_compile_args`` and
``make_thunk``. This makes it add the correct g++ flags to compile with
OpenMP. It also disables OpenMP and prints a warning if the version of
g++ does not support it.

The Aesara flag ``openmp`` is currently False by default as we do not
have code that gets sped up with it. The only current implementation
is ConvOp. It speeds up some cases, but slows down others. That is why
we disable it by default. But we have all the code to have it enabled
by default if there is more than 1 core and the environment
variable OMP_NUM_THREADS is not 1. This allows Aesara to respect the
current convention.

.. note:

   The OpenMP parameter of an Op should not be used in its __eq__ and
   __hash__ methods. Those methods are used to merge equivalent
   computation in an Aesara graph. If we have 2 Apply nodes with the
   same inputs and they execute 2 ConvOp that only differ on the
   OpenMP parameter, we want them to be merged.

.. _numba_ops:

Numba Ops
=========

Want C speed without writing C code for your new Op? You can use Numba
to generate the C code for you! Here is an `example
Op <https://gist.github.com/nouiz/5492778#file-theano_op-py>`_ doing that.

.. _alternate_Aesara_types:

Alternate Aesara Types
======================

Most ops in Aesara are used to manipulate tensors. However, Aesara also
supports many other variable types. The supported types are listed below,
along with pointers to the relevant documentation.

*       :class:`TensorType <tensor.type.TensorType>` : Aesara type that represents
        a multidimensional array containing elements that all have the same
        type. Variables of this Aesara type are represented in C as objects of
        class
        `PyArrayObject <http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#PyArrayObject>`_.

*       :ref:`TypedList <libdoc_typed_list>` : Aesara type that represents a
        typed list (a list where every element in the list has the same Aesara
        type). Variables of this Aesara type are represented in C as objects
        of class `PyListObject <https://docs.python.org/2/c-api/list.html>`_.

*       :ref:`ScalarType <libdoc_scalar>` : Aesara type that represents a C
        primitive type. The C type associated with this Aesara type is the
        represented C primitive itself.

*       :ref:`SparseTensorType <sparse_ops>` : Aesara `Type` used to represent sparse
        tensors. There is no equivalent C type for this Aesara `Type` but you
        can split a sparse variable into its parts as TensorVariables. Those
        can then be used as inputs to an op with C code.

*       :class:`Generic <aesara.link.c.type.Generic>` : Aesara type that
        represents a simple Python Object. Variables of this Aesara type are
        represented in C as objects of class `PyObject
        <https://docs.python.org/2/c-api/structures.html#c.PyObject>`_.

*       :class:`CDataType <aesara.link.c.type.CDataType>` :  Aesara type that
        represents a C data type. The C type associated with this Aesara type
        depends on the data being represented.


.. _creating_a_c_op:

=====================================
Extending Aesara with a C :Class:`Op`
=====================================

This tutorial covers how to extend Aesara with an :class:`Op` that offers a C
implementation.  This tutorial is aimed at individuals who already know how to
extend Aesara (see tutorial :ref:`creating_an_op`) by adding a new :class:`Op`
with a Python implementation and will only cover the additional knowledge
required to also produce :class:`Op`\s with C implementations.

Providing an Aesara :class:`Op` with a C implementation requires to interact with
Python's C-API and Numpy's C-API. Thus, the first step of this tutorial is to
introduce both and highlight their features which are most relevant to the
task of implementing a C :class:`Op`. This tutorial then introduces the most important
methods that the :class:`Op` needs to implement in order to provide a usable C
implementation. Finally, it shows how to combine these elements to write a
simple C :class:`Op` for performing the simple task of multiplying every element in a
vector by a scalar.

Python C-API
============

Python provides a C-API to allows the manipulation of python objects from C
code. In this API, all variables that represent Python objects are of type
``PyObject *``. All objects have a pointer to their type object and a reference
count field (that is shared with the python side). Most python methods have
an equivalent C function that can be called on the ``PyObject *`` pointer.

As such, manipulating a PyObject instance is often straight-forward but it
is important to properly manage its reference count. Failing to do so can
lead to undesired behavior in the C code.


Reference counting
------------------

Reference counting is a mechanism for keeping track, for an object, of
the number of references to it held by other entities. This mechanism is often
used for purposes of garbage collecting because it allows to easily see if
an object is still being used by other entities. When the reference count
for an object drops to 0, it means it is not used by anyone any longer and can
be safely deleted.

``PyObject``\s implement reference counting and the Python C-API defines a number
of macros to help manage those reference counts. The definition of these
macros can be found here : `Python C-API Reference Counting
<https://docs.python.org/2/c-api/refcounting.html>`_. Listed below are the
two macros most often used in Aesara C :class:`Op`\s.


.. method:: void Py_XINCREF(PyObject *o)

    Increments the reference count of object ``o``. Without effect if the
    object is NULL.

.. method:: void Py_XDECREF(PyObject *o)

    Decrements the reference count of object ``o``. If the reference count
    reaches 0, it will trigger a call of the object's deallocation function.
    Without effect if the object is NULL.

The general principle, in the reference counting paradigm, is that the owner
of a reference to an object is responsible for disposing properly of it.
This can be done by decrementing the reference count once the reference is no
longer used or by transferring ownership; passing on the reference to a new
owner which becomes responsible for it.

Some functions return "borrowed references"; this means that they return a
reference to an object **without** transferring ownership of the reference to the
caller of the function. This means that if you call a function which returns a
borrowed reference, you do not have the burden of properly disposing of that
reference. You should **not** call Py_XDECREF() on a borrowed reference.

Correctly managing the reference counts is important as failing to do so can
lead to issues ranging from memory leaks to segmentation faults.


NumPy C-API
===========

The NumPy library provides a C-API to allow users to create, access and
manipulate NumPy arrays from within their own C routines. NumPy's :class:`ndarray`\s
are used extensively inside Aesara and so extending Aesara with a C :class:`Op` will
require interaction with the NumPy C-API.

This sections covers the API's elements that are often required to write code
for an Aesara C :class:`Op`. The full documentation for the API can be found here :
`NumPy C-API <http://docs.scipy.org/doc/numpy/reference/c-api.html>`_.


NumPy data types
----------------

To allow portability between platforms, the NumPy C-API defines its own data
types which should be used whenever you are manipulating a NumPy array's
internal data. The data types most commonly used to implement C :class:`Op`\s are the
following : ``npy_int{8,16,32,64}``, ``npy_uint{8,16,32,64}`` and
``npy_float{32,64}``.

You should use these data types when manipulating a NumPy array's internal
data instead of C primitives because the size of the memory representation
for C primitives can vary between platforms. For instance, a C ``long`` can be
represented in memory with 4 bytes but it can also be represented with 8.
On the other hand, the in-memory size of NumPy data types remains constant
across platforms. Using them will make your code simpler and more portable.

The full list of defined data types can be found here :
`NumPy C-API data types
<http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html#c-type-names>`_.


NumPy :class:`ndarray`\s
------------------------

In the NumPy C-API, NumPy arrays are represented as instances of the
PyArrayObject class which is a descendant of the PyObject class. This means
that, as for any other Python object that you manipulate from C code, you
need to appropriately manage the reference counts of PyArrayObject instances.

Unlike in a standard multidimensional C array, a NumPy array's internal data
representation does not have to occupy a continuous region in memory. In fact,
it can be C-contiguous, F-contiguous or non-contiguous. C-contiguous means
that the data is not only contiguous in memory but also that it is organized
such that the index of the latest dimension changes the fastest. If the
following array

.. testcode::

    x = [[1, 2, 3],
         [4, 5, 6]]

is C-contiguous, it means that, in memory, the six values contained in the
array ``x`` are stored in the order ``[1, 2, 3, 4, 5, 6]`` (the first value is
``x[0,0]``, the second value is ``x[0,1]``, the third value is ``x[0,2]``, the,
fourth value is ``x[1,0]``, etc). F-contiguous (or Fortran Contiguous) also
means that the data is contiguous but that it is organized such that the index
of the latest dimension changes the slowest. If the array ``x`` is
F-contiguous, it means that, in memory, the values appear in the order
``[1, 4, 2, 5, 3, 6]`` (the first value is ``x[0,0]``, the second value is
``x[1,0]``, the third value is ``x[0,1]``, etc).

Finally, the internal data can be non-contiguous. In this case, it occupies
a non-contiguous region in memory but it is still stored in an organized
fashion : the distance between the element ``x[i,j]`` and the element
``x[i+1,j]`` of the array is constant over all valid values of ``i`` and
``j``, just as the distance between the element ``x[i,j]`` and the element
``x[i,j+1]`` of the array is constant over all valid values of ``i`` and ``j``.
This distance between consecutive elements of an array over a given dimension,
is called the stride of that dimension.


Accessing NumPy :class`ndarray`\s' data and properties
------------------------------------------------------

The following macros serve to access various attributes of NumPy :class:`ndarray`\s.

.. method:: void* PyArray_DATA(PyArrayObject* arr)

    Returns a pointer to the first element of the array's data. The returned
    pointer must be cast to a pointer of the proper Numpy C-API data type
    before use.

.. method:: int PyArray_NDIM(PyArrayObject* arr)

    Returns the number of dimensions in the the array pointed by ``arr``

.. method:: npy_intp* PyArray_DIMS(PyArrayObject* arr)

    Returns a pointer on the first element of ``arr``'s internal array
    describing its dimensions. This internal array contains as many elements
    as the array ``arr`` has dimensions.

    The macro ``PyArray_SHAPE()`` is a synonym of ``PyArray_DIMS()`` : it has
    the same effect and is used in an identical way.

.. method:: npy_intp* PyArray_STRIDES(PyArrayObject* arr)

    Returns a pointer on the first element of ``arr``'s internal array
    describing the stride for each of its dimension. This array has as many
    elements as the number of dimensions in ``arr``. In this array, the
    strides are expressed in number of bytes.

.. method:: PyArray_Descr* PyArray_DESCR(PyArrayObject* arr)

    Returns a reference to the object representing the dtype of the array.

    The macro ``PyArray_DTYPE()`` is a synonym of the ``PyArray_DESCR()`` : it
    has the same effect and is used in an identical way.

    :note:
        This is a borrowed reference so you do not need to decrement its
        reference count once you are done with it.

.. method:: int PyArray_TYPE(PyArrayObject* arr)

    Returns the typenumber for the elements of the array. Like the dtype, the
    typenumber is a descriptor for the type of the data in the array. However,
    the two are not synonyms and, as such, cannot be used in place of the
    other.

.. method:: npy_intp PyArray_SIZE(PyArrayObject* arr)

    Returns to total number of elements in the array

.. method:: bool PyArray_CHKFLAGS(PyArrayObject* arr, flags)

    Returns true if the array has the specified flags. The variable flag
    should either be a NumPy array flag or an integer obtained by applying
    bitwise or to an ensemble of flags.

    The flags that can be used in with this macro are :
    ``NPY_ARRAY_C_CONTIGUOUS``, ``NPY_ARRAY_F_CONTIGUOUS``, ``NPY_ARRAY_OWNDATA``,
    ``NPY_ARRAY_ALIGNED``, ``NPY_ARRAY_WRITEABLE``, ``NPY_ARRAY_UPDATEIFCOPY``.


Creating NumPy :class:`ndarray`\s
---------------------------------

The following functions allow the creation and copy of NumPy arrays :

.. method:: PyObject* PyArray_EMPTY(int nd, npy_intp* dims, typenum dtype,
                                    int fortran)

    Constructs a new :class:`ndarray` with the number of dimensions specified by
    ``nd``, shape specified by ``dims`` and data type specified by ``dtype``.
    If ``fortran`` is equal to 0, the data is organized in a C-contiguous
    layout, otherwise it is organized in a F-contiguous layout. The array
    elements are not initialized in any way.

    The function ``PyArray_Empty()`` performs the same function as the macro
    ``PyArray_EMPTY()`` but the data type is given as a pointer to a
    ``PyArray_Descr`` object instead of a ``typenum``.

.. method:: PyObject* PyArray_ZEROS(int nd, npy_intp* dims, typenum dtype,
                                    int fortran)

    Constructs a new :class:`ndarray` with the number of dimensions specified by
    ``nd``, shape specified by ``dims`` and data type specified by ``dtype``.
    If ``fortran`` is equal to 0, the data is organized in a C-contiguous
    layout, otherwise it is organized in a F-contiguous layout. Every element
    in the array is initialized to 0.

    The function ``PyArray_Zeros()`` performs the same function as the macro
    ``PyArray_ZEROS()`` but the data type is given as a pointer to a
    ``PyArray_Descr`` object instead of a ``typenum``.

.. method:: PyArrayObject* PyArray_GETCONTIGUOUS(PyObject* op)

    Returns a C-contiguous and well-behaved copy of the array :class:`Op`. If :class:`Op` is
    already C-contiguous and well-behaved, this function simply returns a
    new reference to :class:`Op`.



Methods the C :Class:`Op` needs to define
=========================================

There is a key difference between an :class:`Op` defining a Python implementation for
its computation and defining a C implementation. In the case of a Python
implementation, the :class:`Op` defines a function ``perform()`` which executes the
required Python code to realize the :class:`Op`. In the case of a C implementation,
however, the :class:`Op` does **not** define a function that will execute the C code; it
instead defines functions that will **return** the C code to the caller.

This is because calling C code from Python code comes with a significant
overhead. If every :class:`Op` was responsible for executing its own C code, every
time an Aesara function was called, this overhead would occur as many times
as the number of :class:`Op`\s with C implementations in the function's computational
graph.

To maximize performance, Aesara instead requires the C :class:`Op`\s to simply return
the code needed for their execution and takes upon itself the task of
organizing, linking and compiling the code from the various :class:`Op`\s. Through this,
Aesara is able to minimize the number of times C code is called from Python
code.

The following is a very simple example to illustrate how it's possible to
obtain performance gains with this process. Suppose you need to execute,
from Python code, 10 different :class:`Op`\s, each one having a C implementation. If
each :class:`Op` was responsible for executing its own C code, the overhead of
calling C code from Python code would occur 10 times. Consider now the case
where the :class:`Op`\s instead return the C code for their execution. You could get
the C code from each :class:`Op` and then define your own C module that would call
the C code from each :class:`Op` in succession. In this case, the overhead would only
occur once; when calling your custom module itself.

Moreover, the fact that Aesara itself takes care of compiling the C code,
instead of the individual :class:`Op`\s, allows Aesara to easily cache the compiled C
code. This allows for faster compilation times.

The following are some of the various methods of the class :class:`COp` that are
related to the C implementation:

* The methods :meth:`CLinkerObject.c_libraries` and :meth:`CLinkerObject.c_lib_dirs` to allow
  your :class:`Op` to use external libraries.

* The method :meth:`CLinkerOp.c_code_cleanup` to specify how the :class:`Op` should
  clean up what it has allocated during its execution.

* The methods :meth:`COp.c_init_code` and :meth:`CLinkerOp.c_init_code_apply`
  to specify code that should be executed once when the module is
  initialized, before anything else is executed.

* The methods :meth:`CLinkerObject.c_compile_args` and
  :meth:`CLinkerObject.c_no_compile_args` to specify requirements regarding how
  the :class:`Op`'s C code should be compiled.

This section describes the methods :meth:`CLinkerOp.c_code`,
:meth:`CLinkerObject.c_support_code`, :meth:`Op.c_support_code_apply` and
:meth:`CLinkerObject.c_code_cache_version` because they are the ones that are most
commonly used.

.. method:: c_code(node, name, input_names, output_names, sub)

    This method returns a string containing the C code to perform the
    computation required by this `Op`.

    The ``node`` argument is an :ref:`apply` node representing an
    application of the current `Op` on a list of inputs, producing a list of
    outputs.

    ``input_names`` is a sequence of strings which contains as many strings
    as the `Op` has inputs. Each string contains the name of the C variable
    to which the corresponding input has been assigned. For example, the name
    of the C variable representing the first input of the `Op` is given by
    ``input_names[0]``. You should therefore use this name in your
    C code to interact with that variable. ``output_names`` is used
    identically to ``input_names``, but for the `Op`'s outputs.

    Finally, ``sub`` is a dictionary of extras parameters to the `c_code`
    method. Among other things, it contains ``sub['fail']`` which is a string
    of C code that you should include in your C code (after ensuring that a
    Python exception is set) if it needs to raise an exception. Ex:

    .. code-block:: c

        c_code = """
            PyErr_Format(PyExc_ValueError, "X does not have the right value");
            %(fail)s;
        """ % {'fail' : sub['fail']}

    to raise a ValueError Python exception with the specified message.
    The function ``PyErr_Format()`` supports string formatting so it is
    possible to tailor the error message to the specifics of the error
    that occurred. If ``PyErr_Format()`` is called with more than two
    arguments, the subsequent arguments are used to format the error message
    with the same behavior as the function `PyString_FromFormat()
    <https://docs.python.org/2/c-api/string.html#c.PyString_FromFormat>`_. The
    ``%`` characters in the format characters need to be escaped since the C
    code itself is defined in a string which undergoes string formatting.

    .. code-block:: c

        c_code = """
            PyErr_Format(PyExc_ValueError,
                         "X==%%i but it should be greater than 0", X);
            %(fail)s;
        """ % {'fail' : sub['fail']}

    :note:
        Your C code should not return the output of the computation but
        rather put the results in the C variables whose names are contained in
        the ``output_names``.

.. method:: c_support_code(**kwargs)

    Returns a string or a list of strings containing some support C code for this `Op`. This code
    will be included at the global scope level and can be used to define
    functions and structs that will be used by every apply of this `Op`.

.. method:: c_support_code_apply(node, name)

    Returns a string containing some support C code for this `Op`. This code
    will be included at the global scope level and can be used to define
    functions and structs that will be used by this `Op`. The difference between
    this method and ``c_support_code`` is that the C code specified in
    ``c_support_code_apply`` should be specific to each apply of the `Op`,
    while ``c_support_code`` is for support code that is not specific to
    each apply.

    Both ``c_support_code`` and ``c_support_code_apply`` are necessary
    because an Aesara `Op` can be used more than once in a given Aesara
    function. For example, an `Op` that adds two matrices could be used at some
    point in the Aesara function to add matrices of integers and, at another
    point, to add matrices of doubles. Because the dtype of the inputs and
    outputs can change between different applies of the `Op`, any support code
    that relies on a certain dtype is specific to a given `Apply` of the `Op` and
    should therefore be defined in ``c_support_code_apply``.

.. method:: c_code_cache_version()

    Returns a tuple of integers representing the version of the C code in this
    :class:`Op`. Ex : (1, 4, 0) for version 1.4.0

    This tuple is used by Aesara to cache the compiled C code for this `Op`. As
    such, the return value **MUST BE CHANGED** every time the C code is altered
    or else Aesara will disregard the change in the code and simply load a
    previous version of the `Op` from the cache. If you want to avoid caching of
    the C code of this `Op`, return an empty tuple or do not implement this
    method.

    :note:
        Aesara can handle tuples of any hashable objects as return values
        for this function but, for greater readability and easier management,
        this function should return a tuple of integers as previously
        described.

        Also, do not use the built-in ``hash``; it will produce different values
        between Python sessions and confound the caching process.

Important restrictions when implementing a :class:`COp`
=======================================================

There are some important restrictions to remember when implementing an `COp`.
Unless your `COp` correctly defines a ``view_map`` attribute, the ``perform`` and ``c_code`` must not
produce outputs whose memory is aliased to any input (technically, if changing the
output could change the input object in some sense, they are aliased).
Unless your `COp` correctly defines a ``destroy_map`` attribute, ``perform`` and ``c_code`` must
not modify any of the inputs.

TODO: EXPLAIN DESTROYMAP and VIEWMAP BETTER AND GIVE EXAMPLE.

When developing a `COp`, you should run computations in `DebugMode`, by using
argument ``mode='DebugMode'`` to ``aesara.function``. `DebugMode` is
slow, but it can catch many common violations of the `Op` contract.

TODO: Like what? How? Talk about Python vs. C too.

`DebugMode` is no silver bullet though.
For example, if you modify an `Op` ``self.*`` during any of
``make_node``, ``perform``, or ``c_code``, you are probably doing something
wrong but DebugMode will not detect this.

TODO: jpt: I don't understand the following sentence.

`Op`\s and `Type`\s should usually be considered immutable -- you should
definitely not make a change that would have an impact on ``__eq__``,
``__hash__``, or the mathematical value that would be computed by  ``perform``
or ``c_code``.


Simple :class:`COp` example
===========================

In this section, we put together the concepts that were covered in this
tutorial to generate an :class:`Op` which multiplies every element in a vector
by a scalar and returns the resulting vector. This is intended to be a simple
example so the methods ``c_support_code`` and ``c_support_code_apply`` are
not used because they are not required.

In the C code below notice how the reference count on the output variable is
managed. Also take note of how the new variables required for the :class:`Op`'s
computation are declared in a new scope to avoid cross-initialization errors.

Also, in the C code, it is very important to properly validate the inputs
and outputs storage. Aesara guarantees that the inputs exist and have the
right number of dimensions but it does not guarantee their exact shape. For
instance, if an :class:`Op` computes the sum of two vectors, it needs to validate that
its two inputs have the same shape. In our case, we do not need to validate
the exact shapes of the inputs because we don't have a need that they match
in any way.

For the outputs, things are a little bit more subtle. Aesara does not
guarantee that they have been allocated but it does guarantee that, if they
have been allocated, they have the right number of dimension. Again, Aesara
offers no guarantee on the exact shapes. This means that, in our example, we
need to validate that the output storage has been allocated and has the same
shape as our vector input. If it is not the case, we allocate a new output
storage with the right shape and number of dimensions.

.. testcode:: examples

    import numpy
    import aesara

    from aesara.link.c.op import COp
    from aesara.graph.basic import Apply


    class VectorTimesScalar(COp):
        __props__ = ()

        def make_node(self, x, y):
            # Validate the inputs' type
            if x.type.ndim != 1:
                raise TypeError('x must be a 1-d vector')
            if y.type.ndim != 0:
                raise TypeError('y must be a scalar')

            # Create an output variable of the same type as x
            output_var = x.type()

            return Apply(self, [x, y], [output_var])

        def c_code_cache_version(self):
            return (1, 0)

        def c_code(self, node, name, inp, out, sub):
            x, y = inp
            z, = out

            # Extract the dtypes of the inputs and outputs storage to
            # be able to declare pointers for those dtypes in the C
            # code.
            dtype_x = node.inputs[0].dtype
            dtype_y = node.inputs[1].dtype
            dtype_z = node.outputs[0].dtype

            itemsize_x = numpy.dtype(dtype_x).itemsize
            itemsize_z = numpy.dtype(dtype_z).itemsize

            fail = sub['fail']

            c_code = """
            // Validate that the output storage exists and has the same
            // dimension as x.
            if (NULL == %(z)s ||
                PyArray_DIMS(%(x)s)[0] != PyArray_DIMS(%(z)s)[0])
            {
                /* Reference received to invalid output variable.
                Decrease received reference's ref count and allocate new
                output variable */
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*)PyArray_EMPTY(1,
                                                    PyArray_DIMS(%(x)s),
                                                    PyArray_TYPE(%(x)s),
                                                    0);

                if (!%(z)s) {
                    %(fail)s;
                }
            }

            // Perform the vector multiplication by a scalar
            {
                /* The declaration of the following variables is done in a new
                scope to prevent cross initialization errors */
                npy_%(dtype_x)s* x_data_ptr =
                                (npy_%(dtype_x)s*)PyArray_DATA(%(x)s);
                npy_%(dtype_z)s* z_data_ptr =
                                (npy_%(dtype_z)s*)PyArray_DATA(%(z)s);
                npy_%(dtype_y)s y_value =
                                ((npy_%(dtype_y)s*)PyArray_DATA(%(y)s))[0];
                int x_stride = PyArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
                int z_stride = PyArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
                int x_dim = PyArray_DIMS(%(x)s)[0];

                for(int i=0; i < x_dim; i++)
                {
                    z_data_ptr[i * z_stride] = (x_data_ptr[i * x_stride] *
                                                y_value);
                }
            }
            """

            return c_code % locals()


The ``c_code`` method accepts variable names as arguments (``name``, ``inp``,
``out``, ``sub``) and returns a C code fragment that computes the expression
output. In case of error, the ``%(fail)s`` statement cleans up and returns
properly.

More complex C :Class:`Op` example
==================================

This section introduces a new example, slightly more complex than the previous
one, with an :class:`Op` to perform an element-wise multiplication between the elements
of two vectors. This new example differs from the previous one in its use
of the methods ``c_support_code`` and ``c_support_code_apply`` (it does
not `need` to use them but it does so to explain their use) and its capacity
to support inputs of different dtypes.

Recall the method ``c_support_code`` is meant to produce code that will
be used for every apply of the :class:`Op`. This means that the C code in this
method must be valid in every setting your :class:`Op` supports. If the :class:`Op` is meant
to supports inputs of various dtypes, the C code in this method should be
generic enough to work with every supported dtype. If the :class:`Op` operates on
inputs that can be vectors or matrices, the C code in this method should
be able to accommodate both kinds of inputs.

In our example, the method ``c_support_code`` is used to declare a C
function to validate that two vectors have the same shape. Because our
:class:`Op` only supports vectors as inputs, this function is allowed to rely
on its inputs being vectors. However, our :class:`Op` should support multiple
dtypes so this function cannot rely on a specific dtype in its inputs.

The method ``c_support_code_apply``, on the other hand, is allowed
to depend on the inputs to the :class:`Op` because it is apply-specific. Therefore, we
use it to define a function to perform the multiplication between two vectors.
Variables or functions defined in the method ``c_support_code_apply`` will
be included at the global scale for every apply of the :Class:`Op`. Because of this,
the names of those variables and functions should include the name of the :class:`Op`,
like in the example. Otherwise, using the :class:`Op` twice in the same graph will give
rise to conflicts as some elements will be declared more than once.

The last interesting difference occurs in the ``c_code()`` method. Because the
dtype of the output is variable and not guaranteed to be the same as any of
the inputs (because of the upcast in the method ``make_node()``), the typenum
of the output has to be obtained in the Python code and then included in the
C code.

.. testcode:: examples

    class VectorTimesVector(COp):
        __props__ = ()

        def make_node(self, x, y):
            # Validate the inputs' type
            if x.type.ndim != 1:
                raise TypeError('x must be a 1-d vector')
            if y.type.ndim != 1:
                raise TypeError('y must be a 1-d vector')

            # Create an output variable of the same type as x
            output_var = aesara.tensor.type.TensorType(
                            dtype=aesara.scalar.upcast(x.dtype, y.dtype),
                            shape=(None,))()

            return Apply(self, [x, y], [output_var])

        def c_code_cache_version(self):
            return (1, 0, 2)

        def c_support_code(self, **kwargs):
            c_support_code = """
            bool vector_same_shape(PyArrayObject* arr1,
                PyArrayObject* arr2)
            {
                return (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0]);
            }
            """

            return c_support_code

        def c_support_code_apply(self, node, name):
            dtype_x = node.inputs[0].dtype
            dtype_y = node.inputs[1].dtype
            dtype_z = node.outputs[0].dtype

            c_support_code = """
            void vector_elemwise_mult_%(name)s(npy_%(dtype_x)s* x_ptr,
                int x_str, npy_%(dtype_y)s* y_ptr, int y_str,
                npy_%(dtype_z)s* z_ptr, int z_str, int nbElements)
            {
                for (int i=0; i < nbElements; i++){
                    z_ptr[i * z_str] = x_ptr[i * x_str] * y_ptr[i * y_str];
                }
            }
            """

            return c_support_code % locals()

        def c_code(self, node, name, inp, out, sub):
            x, y = inp
            z, = out

            dtype_x = node.inputs[0].dtype
            dtype_y = node.inputs[1].dtype
            dtype_z = node.outputs[0].dtype

            itemsize_x = numpy.dtype(dtype_x).itemsize
            itemsize_y = numpy.dtype(dtype_y).itemsize
            itemsize_z = numpy.dtype(dtype_z).itemsize

            typenum_z = numpy.dtype(dtype_z).num

            fail = sub['fail']

            c_code = """
            // Validate that the inputs have the same shape
            if ( !vector_same_shape(%(x)s, %(y)s))
            {
                PyErr_Format(PyExc_ValueError, "Shape mismatch : "
                            "x.shape[0] and y.shape[0] should match but "
                            "x.shape[0] == %%i and y.shape[0] == %%i",
                            PyArray_DIMS(%(x)s)[0], PyArray_DIMS(%(y)s)[0]);
                %(fail)s;
            }

            // Validate that the output storage exists and has the same
            // dimension as x.
            if (NULL == %(z)s || !(vector_same_shape(%(x)s, %(z)s)))
            {
                /* Reference received to invalid output variable.
                Decrease received reference's ref count and allocate new
                output variable */
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*)PyArray_EMPTY(1,
                                                    PyArray_DIMS(%(x)s),
                                                    %(typenum_z)s,
                                                    0);

                if (!%(z)s) {
                    %(fail)s;
                }
            }

            // Perform the vector elemwise multiplication
            vector_elemwise_mult_%(name)s(
                                    (npy_%(dtype_x)s*)PyArray_DATA(%(x)s),
                                    PyArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                                    (npy_%(dtype_y)s*)PyArray_DATA(%(y)s),
                                    PyArray_STRIDES(%(y)s)[0] / %(itemsize_y)s,
                                    (npy_%(dtype_z)s*)PyArray_DATA(%(z)s),
                                    PyArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                                    PyArray_DIMS(%(x)s)[0]);
            """

            return c_code % locals()


Alternate way of defining C :class:`Op`\s
=========================================

The two previous examples have covered the standard way of implementing C :class:`Op`\s
in Aesara by inheriting from the class :class:`Op`. This process is mostly
simple but it still involves defining many methods as well as mixing, in the
same file, both Python and C code which tends to make the result less
readable.

To help with this, Aesara defines a class, `ExternalCOp`, from which new C :class:`Op`\s
can inherit. The class `ExternalCOp` aims to simplify the process of implementing
C :class:`Op`\s by doing the following :

*      It allows you to define the C implementation of your :class:`Op` in a distinct
       C code file. This makes it easier to keep your Python and C code
       readable and well indented.

*      It can automatically handle all the methods that return C code,
       in addition to :meth:`Op.c_code_cache_version` based on the
       provided external C implementation.

To illustrate how much simpler the class `ExternalCOp` makes the process of defining
a new :class:`Op` with a C implementation, let's revisit the second example of this
tutorial, the `VectorTimesVector`\ :class:`Op`. In that example, we implemented an :class:`Op`
to perform the task of element-wise vector-vector multiplication. The two
following blocks of code illustrate what the :class:`Op` would look like if it was
implemented using the `ExternalCOp` class.

The new :class:`Op` is defined inside a Python file with the following code :

.. testcode::

    import aesara
    from aesara.link.c.op import ExternalCOp

    class VectorTimesVector(ExternalCOp):
        __props__ = ()

        func_file = "./vectorTimesVector.c"
        func_name = "APPLY_SPECIFIC(vector_times_vector)"

        def __init__(self):
            super().__init__(self.func_file, self.func_name)

        def make_node(self, x, y):
            # Validate the inputs' type
            if x.type.ndim != 1:
                raise TypeError('x must be a 1-d vector')
            if y.type.ndim != 1:
                raise TypeError('y must be a 1-d vector')

            # Create an output variable of the same type as x
            output_var = aesara.tensor.type.TensorType(
                            dtype=aesara.scalar.upcast(x.dtype, y.dtype),
                            shape=(None,))()

            return Apply(self, [x, y], [output_var])

And the following is the C implementation of the :class:`Op`, defined in an external
C file named ``vectorTimesVector.c``:

.. code-block:: c

    #section support_code

    // Support code function
    bool vector_same_shape(PyArrayObject* arr1, PyArrayObject* arr2)
    {
        return (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0]);
    }


    #section support_code_apply

    // Apply-specific support function
    void APPLY_SPECIFIC(vector_elemwise_mult)(
        DTYPE_INPUT_0* x_ptr, int x_str,
        DTYPE_INPUT_1* y_ptr, int y_str,
        DTYPE_OUTPUT_0* z_ptr, int z_str, int nbElements)
    {
        for (int i=0; i < nbElements; i++){
            z_ptr[i * z_str] = x_ptr[i * x_str] * y_ptr[i * y_str];
        }
    }

    // Apply-specific main function
    int APPLY_SPECIFIC(vector_times_vector)(PyArrayObject* input0,
                                            PyArrayObject* input1,
                                            PyArrayObject** output0)
    {
        // Validate that the inputs have the same shape
        if ( !vector_same_shape(input0, input1))
        {
            PyErr_Format(PyExc_ValueError, "Shape mismatch : "
                        "input0.shape[0] and input1.shape[0] should "
                        "match but x.shape[0] == %i and "
                        "y.shape[0] == %i",
                        PyArray_DIMS(input0)[0], PyArray_DIMS(input1)[0]);
            return 1;
        }

        // Validate that the output storage exists and has the same
        // dimension as x.
        if (NULL == *output0 || !(vector_same_shape(input0, *output0)))
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(*output0);
            *output0 = (PyArrayObject*)PyArray_EMPTY(1,
                                                    PyArray_DIMS(input0),
                                                    TYPENUM_OUTPUT_0,
                                                    0);

            if (!*output0) {
                PyErr_Format(PyExc_ValueError,
                            "Could not allocate output storage");
                return 1;
            }
        }

        // Perform the actual vector-vector multiplication
        APPLY_SPECIFIC(vector_elemwise_mult)(
                                (DTYPE_INPUT_0*)PyArray_DATA(input0),
                                PyArray_STRIDES(input0)[0] / ITEMSIZE_INPUT_0,
                                (DTYPE_INPUT_1*)PyArray_DATA(input1),
                                PyArray_STRIDES(input1)[0] / ITEMSIZE_INPUT_1,
                                (DTYPE_OUTPUT_0*)PyArray_DATA(*output0),
                                PyArray_STRIDES(*output0)[0] / ITEMSIZE_OUTPUT_0,
                                PyArray_DIMS(input0)[0]);

        return 0;
    }

As you can see from this example, the Python and C implementations are nicely
decoupled which makes them much more readable than when they were intertwined
in the same file and the C code contained string formatting markers.

Now that we have motivated the `ExternalCOp` class, we can have a more precise look at
what it does for us. For this, we go through the various elements that make up
this new version of the `VectorTimesVector`\ `Op` :

*       Parent class : instead of inheriting from the class :class:`Op`,
        VectorTimesVector inherits from the class `ExternalCOp`.

*       Constructor : in our new `COp`, the :meth:`COp.__init__` method has an
        important use; to inform the constructor of the `ExternalCOp` class
        of the location, on the filesystem of the C implementation of
        this `COp`. To do this, it gives a list of file paths containing
        the C code for this `COp`.  To auto-generate the c_code method
        with a function call you can specify the function name as the
        second parameter.  The paths should be given as a relative
        path from the folder where the descendant of the `ExternalCOp` class
        is defined.

*       :meth:`ExternalCOp.make_node` : this method is absolutely
        identical to the one in our old example. Using the `ExternalCOp`
        class doesn't change anything here.

*       External C code : the external C code implements the various
        functions associated with the `COp`.  Writing this C code
        involves a few subtleties which deserve their own respective
        sections.

Main function
-------------

If you pass a function name to :meth:`ExternalCOp.__init___`, it must respect
the following constraints:

*       It must return an int. The value of that int indicates whether
        the `Op` could perform its task or not. A value of 0 indicates
        success while any non-zero value will interrupt the execution
        of the Aesara function.  When returning non-zero the function
        must set a python exception indicating the details of the
        problem.

*       It must receive one argument for each input to the `Op` followed
        by one pointer to an argument for each output of the `Op`.  The
        types for the argument is dependent on the Types (that is
        aesara Types) of your inputs and outputs.

*       You can specify the number of inputs and outputs for your `Op`
        by setting the ``_cop_num_inputs`` and ``_cop_num_outputs``
        attributes on your `COp`.  The main function will always be
        called with that number of arguments, using NULL to fill in
        for missing values at the end.  This can be used if your `COp`
        has a variable number of inputs or outputs, but with a fixed
        maximum.

For example, the main C function of an `COp` that takes two TensorTypes
(which has ``PyArrayObject *`` as its C type) as inputs and returns
both their sum and the difference between them would have four
parameters (two for the `COp`'s inputs and two for its outputs) and it's
signature would look something like this :

.. code-block:: c

    int sumAndDiffOfScalars(PyArrayObject* in0, PyArrayObject* in1,
                            PyArrayObject** out0, PyArrayObject** out1)

Macros
------

For certain section tags, your C code can benefit from a number of
pre-defined macros.  These section tags have no macros: ``init_code``,
``support_code``. All other tags will have the support macros
discussed below.

*      ``APPLY_SPECIFIC(str)`` which will automatically append a name
       unique to the :ref:`Apply` node that applies the `Op` at the end
       of the provided ``str``. The use of this macro is discussed
       further below.

For every input which has a :attr:`dtype` attribute (this means
Tensors), the following macros will be
defined unless your `Op` class has an :attr:`Op.check_input` attribute
defined to False. In these descrptions 'i' refers to the position
(indexed from 0) in the input array.

*       ``DTYPE_INPUT_{i}`` : NumPy dtype of the data in the array.
        This is the variable type corresponding to the NumPy dtype, not the
        string representation of the NumPy dtype. For instance, if the `Op`'s
        first input is a float32 :class:`ndarray`, then the macro ``DTYPE_INPUT_0``
        corresponds to ``npy_float32`` and can directly be used to declare a
        new variable of the same dtype as the data in the array :

        .. code-block:: c

            DTYPE_INPUT_0 myVar = someValue;

*       ``TYPENUM_INPUT_{i}`` : Typenum of the data in the array

*       ``ITEMSIZE_INPUT_{i}`` : Size, in bytes, of the elements in
        the array.

In the same way, the macros ``DTYPE_OUTPUT_{i}``,
``ITEMSIZE_OUTPUT_{i}`` and ``TYPENUM_OUTPUT_{i}`` are defined for
every output 'i' of the `Op`.

In addition to these macros, the ``init_code_struct``, ``code``, and
``code_cleanup`` section tags also have the following macros:

*     ``FAIL`` : Code to insert at error points.  A python exception
      should be set prior to this code.  An invocation look like this:

      .. code-block:: c

        if (error) {
          // Set python exception
          FAIL
        }

      You can add a semicolon after the macro if it makes your editor
      happy.

*     ``PARAMS`` : Name of the params variable for this node.  (only
      for `Op`\s which have params, which is discussed elsewhere)

Finally the tag ``code`` and ``code_cleanup`` have macros to
pass the inputs and output names.  These are name ``INPUT_{i}`` and
``OUTPUT_{i}`` where `i` is the 0-based index position in the input
and output arrays respectively.

Support code
------------

Certain section are limited in what you can place in them due to
semantic and syntactic restrictions of the C++ language.  Most of
these restrictions apply to the tags that end in ``_struct``.

When we defined the ``VectorTimesVector`` `Op` without using the ``ExternalCOp``
class, we had to make a distinction between two types of support_code
: the support code that was apply-specific and the support code that
wasn't. The apply-specific code was defined in the
``c_support_code_apply`` method and the elements defined in that
code (global variables and functions) had to include the name of the
Apply node in their own names to avoid conflicts between the different
versions of the apply-specific code. The code that wasn't
apply-specific was simply defined in the ``c_support_code`` method.

To make indentifiers that include the :ref:`Apply` node name use the
``APPLY_SPECIFIC(str)`` macro. In the above example, this macro is
used when defining the functions ``vector_elemwise_mult`` and
``vector_times_vector`` as well as when calling function
``vector_elemwise_mult`` from inside ``vector_times_vector``.

When using the ``ExternalCOp`` class, we still have to make the distinction
between C code for each of the methods of a C class. These sections of
code are separated by ``#section <tag>`` markers. The tag determines
the name of the method this C code applies to with the rule that
``<tag>`` applies to `c_<tag>`. Unknown tags are an error and will be
reported. Duplicate tags will be merged together in the order the
appear in the C files.

The rules for knowing if where a piece of code should be put can be
sometimes tricky.  The key thing to remember is that things that can
be shared between instances of the `Op` should be apply-agnostic and go
into a section which does not end in ``_apply`` or ``_struct``.  The
distinction of ``_apply`` and ``_struct`` mostly hinghes on how you
want to manage the lifetime of the object.  Note that to use an
apply-specific object, you have to be in a apply-specific section, so
some portions of the code that might seem apply-agnostic may still be
apply-specific because of the data they use (this does not include
arguments).

In the above example, the ``function vector_same_shape`` is
apply-agnostic because it uses none of the macros defined by the class
``ExternalCOp`` and it doesn't rely on any apply-specific code. The function
``vector_elemwise_mult`` is apply-specific because it uses the
macros defined by ``ExternalCOp``. Finally, the function
``vector_times_vector`` is apply-specific because it uses those same
macros and also because it calls ``vector_elemwise_mult`` which is
an apply-specific function.


Using GDB to debug :class:`COp`'s C code
========================================

When debugging C code, it can be useful to use GDB for code compiled
by Aesara.

For this, you must enable this Aesara: `cmodule__remove_gxx_opt=True`.

Then you must start Python inside GDB and in it start your Python
process:

.. code-block:: sh

    $gdb python
    (gdb)r pytest aesara/

`Quick guide to GDB <https://www.cs.cmu.edu/~gilpin/tutorial/>`_.

Final Note
==========

This tutorial focuses on providing C implementations to `COp`s that manipulate
Aesara tensors. For more information about other Aesara types, you can refer
to the section :ref:`Alternate Aesara Types <alternate_aesara_types>`.

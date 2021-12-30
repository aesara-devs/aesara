Adding JAX and Numba support for `Op`\s
=======================================

Aesara is able to convert its graphs into JAX and Numba compiled functions. In order to do
this, each :class:`Op` in an Aesara graph must have an equivalent JAX/Numba implementation function.

This tutorial will explain how JAX and Numba implementations are created for an :class:`Op`.  It will
focus specifically on the JAX case, but the same mechanisms are used for Numba as well.

Step 1: Identify the Aesara :class:`Op` you’d like to implement in JAX
----------------------------------------------------------------------

Find the source for the Aesara :class:`Op` you’d like to be supported in JAX, and
identify the function signature and return values.  These can be determined by
looking at the :meth:`Op.make_node` implementation.  In general, one needs to be familiar
with Aesara :class:`Op`\s in order to provide a conversion implementation, so first read
:ref:`creating_an_op` if you are not familiar.

For example, the :class:`Eye`\ :class:`Op` current has an :meth:`Op.make_node` as follows:

.. code:: python

    def make_node(self, n, m, k):
        n = as_tensor_variable(n)
        m = as_tensor_variable(m)
        k = as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        return Apply(
            self,
            [n, m, k],
            [TensorType(dtype=self.dtype, shape=(False, False))()],
        )


The :class:`Apply` instance that's returned specifies the exact types of inputs that
our JAX implementation will receive and the exact types of outputs it's expected to
return--both in terms of their data types and number of dimensions.
The actual inputs our implementation will receive are necessarily numeric values
or NumPy :class:`ndarray`\s; all that :meth:`Op.make_node` tells us is the
general signature of the underlying computation.

More specifically, the :class:`Apply` implies that the inputs come from values that are
automatically converted to Aesara variables via :func:`as_tensor_variable`, and
the ``assert``\s that follow imply that they must be scalars.  According to this
logic, the inputs could have any data type (e.g. floats, ints), so our JAX
implementation must be able to handle all the possible data types.

It also tells us that there's only one return value, that it has a data type
determined by :attr:`Eye.dtype`, and that it has two non-broadcastable
dimensions.  The latter implies that the result is necessarily a matrix.  The
former implies that our JAX implementation will need to access the :attr:`dtype`
attribute of the Aesara :class:`Eye`\ :class:`Op` it's converting.

Next, we can look at the :meth:`Op.perform` implementation to see exactly
how the inputs and outputs are used to compute the outputs for an :class:`Op`
in Python.  This method is effectively what needs to be implemented in JAX.


Step 2: Find the relevant JAX method (or something close)
---------------------------------------------------------

With a precise idea of what the Aesara :class:`Op` does we need to figure out how
to implement it in JAX. In the best case scenario, JAX has a similarly named
function that performs exactly the same computations as the :class:`Op`. For
example, the :class:`Eye` operator has a JAX equivalent: :func:`jax.numpy.eye`
(see `the documentation <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html?highlight=eye>`_).

If we wanted to implement an :class:`Op` like :class:`IfElse`, we might need to
recreate the functionality with some custom logic.  In many cases, at least some
custom logic is needed to reformat the inputs and outputs so that they exactly
match the `Op`'s.

Here's an example for :class:`IfElse`:

.. code:: python

   def ifelse(cond, *args, n_outs=n_outs):
       res = jax.lax.cond(
           cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
       )
       return res if n_outs > 1 else res[0]


Step 3: Register the function with the `jax_funcify` dispatcher
---------------------------------------------------------------

With the Aesara `Op` replicated in JAX, we’ll need to register the
function with the Aesara JAX `Linker`. This is done through the use of
`singledispatch`. If you don't know how `singledispatch` works, see the
`Python documentation <https://docs.python.org/3/library/functools.html#functools.singledispatch>`_.

The relevant dispatch functions created by `singledispatch` are :func:`aesara.link.numba.dispatch.numba_funcify` and
:func:`aesara.link.jax.dispatch.jax_funcify`.

Here’s an example for the `Eye`\ `Op`:

.. code:: python

   import jax.numpy as jnp

   from aesara.tensor.basic import Eye
   from aesara.link.jax.dispatch import jax_funcify


   @jax_funcify.register(Eye)
   def jax_funcify_Eye(op):

       # Obtain necessary "static" attributes from the Op being converted
       dtype = op.dtype

       # Create a JAX jit-able function that implements the Op
       def eye(N, M, k):
           return jnp.eye(N, M, k, dtype=dtype)

       return eye


Step 4: Write tests
-------------------

Test that your registered `Op` is working correctly by adding tests to the
appropriate test suites in Aesara (e.g. in ``tests.link.test_jax`` and one of
the modules in ``tests.link.numba.dispatch``). The tests should ensure that your implementation can
handle the appropriate types of inputs and produce outputs equivalent to `Op.perform`.
Check the existing tests for the general outline of these kinds of tests. In
most cases, a helper function can be used to easily verify the correspondence
between a JAX/Numba implementation and its `Op`.

For example, the :func:`compare_jax_and_py` function streamlines the steps
involved in making comparisons with `Op.perform`.

Here's a small example of a test for :class:`Eye`:

.. code:: python

   import aesara.tensor as at

   def test_jax_Eye():
       """Test JAX conversion of the `Eye` `Op`."""

       # Create a symbolic input for `Eye`
       x_at = at.scalar()

       # Create a variable that is the output of an `Eye` `Op`
       eye_var = at.eye(x_at)

       # Create an Aesara `FunctionGraph`
       out_fg = FunctionGraph(outputs=[eye_var])

       # Pass the graph and any inputs to the testing function
       compare_jax_and_py(out_fg, [3])

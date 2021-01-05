Tutorial on adding JAX Ops to Aesara
=============================

A core feature of Aesara, previously named Theano-PyMC, is the JAX
backend. To support the backend JAX ops need be added to Aesara once to
be supported. This tutorial will explain each step.

Step 1: Identify the Aesara Op you’d like to JAXify
===================================================

Determine which Aesara Op you’d like supported with JAX and identify the
function signature and return values. This will come in handy as we need
to know what we want JAX to do.

| Here are the examples for ``eye`` and ``ifelse`` from Aesara from the
  compiled doc and codebase respectively
| https://theano-pymc.readthedocs.io/en/latest/library/tensor/basic.html?highlight=eye#theano.tensor.eye
| https://github.com/pymc-devs/Theano-PyMC/blob/master/theano/ifelse.py#L35

Step 2: Find the relevant JAX method (or something close)
=========================================================

With a precise idea of what the Aesara Op does we need to figure out how
to implement it in JAX. In easiest scenario JAX has a similarly named
method that does the same thing. For example with the ``eye`` operator
we find the paired ``jax.numpy.eye`` method.

https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html?highlight=eye

For ifelse we’ll need to recreate the functionality with some custom
logic.

.. code:: python

   def ifelse(cond, *args, n_outs=n_outs):
       res = jax.lax.cond(
           cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
       )
       return res if n_outs > 1 else res[0]

*Code in context:*
https://github.com/pymc-devs/Theano-PyMC/blob/master/theano/link/jax/jax_dispatch.py#L583

Step 3: Register the function with the jax_funcify dispatcher
=============================================================

With the Aesara Op replicated in JAX we’ll need to now register this
function with the Aesara JAX Linker. This is done through the dispatcher
decorator and closure as seen below. If unsure how dispatching works a
short tutorial on dispatching is at the bottom.

The linker functions should be added to ``jax_dispatch`` module linked
below.
https://github.com/pymc-devs/Theano-PyMC/blob/master/theano/link/jax/jax_dispatch.py

Here’s an example for the Eye Op.

.. code:: python

   from theano.tensor.basic import Eye

   @jax_funcify.register(Eye) # The decorator
   def jax_funcify_Eye(op): # The function that takes an Op and returns its JAX equivalent
       dtype = op.dtype

       def eye(N, M, k):
           return jnp.eye(N, M, k, dtype=dtype)

       return eye

*Code in context:*
https://github.com/pymc-devs/Theano-PyMC/blob/master/theano/link/jax/jax_dispatch.py#L1071

Step 4: Write tests
===================

Test that your registered Op is working correctly by adding a test to
the ``test_jax.py`` test suite. The test should ensure that Aesara Op,
when included as part of a function graph, passes the tests in
``compare_jax_and_py`` test method. What this test method does is
compile the same function graph in Python and JAX and check that the
numerical output is similar betwen the JAX and Python output, as well
object types to ensure correct compilation.

https://github.com/pymc-devs/Theano-PyMC/blob/master/tests/link/test_jax.py

.. code:: python

   def test_jax_eye():
       """Tests jaxification of the Eye operator"""
       out = tt.eye(3) # Initialize a Theano Op
       out_fg = theano.gof.fg.FunctionGraph([], [out]) # Create a Theano FunctionGraph

       compare_jax_and_py(out_fg, []) # Pas the graph and any inputs to testing function

*Code in context:*
https://github.com/pymc-devs/Theano-PyMC/blob/056fcee1434818d0aed9234e01c754ed88d0f27a/tests/link/test_jax.py#L250

Step 5: Wait for CI pass and Code Review
========================================

Create a pull request and ensure CI passes. If it does wait for a code
review and a likely merge!

https://github.com/pymc-devs/Theano-PyMC/pulls

Appendix: What does singledispatcher do?
========================================

In short a dispatcher figures out what “the right thing” is to do based
on the type of the first argument to the function. It’s easiest
explained with an example. One is provided below in addition to the
python docs.

https://docs.python.org/3/library/functools.html#functools.singledispatch

.. code:: ipython3

    from functools import singledispatch

    class Cow:
        pass
    cow = Cow()

    class Dog:
        pass
    dog = Dog()

    @singledispatch
    def greeting(animal):
        print("This animal has not been registered")

    @greeting.register(Cow)
    def cow_greeting(animal):
        print("Mooooo")

    @greeting.register(Dog)
    def dog_greeting(animal):
        print("Woof")


    greeting(cow)
    greeting(dog)
    greeting("A string object")


.. parsed-literal::

    Mooooo
    Woof
    Animal has not been registerd


This is what allows the JAX Linker to determine which the correct
JAXification Op is as we’ve registered it with the Aesara Op

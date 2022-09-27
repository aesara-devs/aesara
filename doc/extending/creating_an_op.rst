
.. _creating_an_op:

Creating a new :class:`Op`: Python implementation
=================================================

So suppose you have looked through the library documentation and you don't see
a function that does what you want.

If you can implement something in terms of an existing :ref:`Op`, you should do that.
Odds are your function that uses existing Aesara expressions is short,
has no bugs, and potentially profits from rewrites that have already been
implemented.

However, if you cannot implement an :class:`Op` in terms of an existing :class:`Op`, you have to
write a new one.

As an illustration, this tutorial will demonstrate how a simple Python-based
:class:`Op` that performs operations on ``np.float64``\s is written.

.. note::

    This is an introductury tutorial and as such it does not cover how to make
    an :class:`Op` that returns a view or modifies the values in its inputs. Thus, all
    :class:`Op`\s created with the instructions described here MUST return newly
    allocated memory or reuse the memory provided in the parameter
    ``output_storage`` of the :meth:`Op.perform` method. See
    :ref:`views_and_inplace` for an explanation on how to do this.

    If your :class:`Op` returns a view or changes the value of its inputs
    without doing as prescribed in that page, Aesara will run, but will
    return correct results for some graphs and wrong results for others.

    It is recommended that you run your tests in :class:`DebugMode`, since it
    can help verify whether or not your :class:`Op` behaves correctly in this
    regard.


Aesara Graphs refresher
-----------------------

.. image:: apply.png
    :width: 500 px

Aesara represents symbolic mathematical computations as graphs. Those graphs
are bi-partite graphs (graphs with two types of nodes), they are composed of
interconnected :ref:`apply` and :ref:`variable` nodes.
:class:`Variable` nodes represent data in the graph, either inputs, outputs or
intermediary values. As such, inputs and outputs of a graph are lists of Aesara
:class:`Variable` nodes. :class:`Apply` nodes perform computation on these
variables to produce new variables. Each :class:`Apply` node has a link to an
instance of :class:`Op` which describes the computation to perform. This tutorial
details how to write such an :class:`Op` instance. Please refers to
:ref:`graphstructures` for a more detailed explanation about the graph
structure.


:class:`Op`'s basic methods
---------------------------

An :class:`Op` is any Python object which inherits from :class:`Op`.
This section provides an overview of the basic methods you typically have to
implement to make a new :class:`Op`.  It does not provide extensive coverage of all the
possibilities you may encounter or need.  For that refer to
:ref:`op_contract`.

.. testcode:: python

    import aesara
    from aesara.graph.op import Op


    class MyOp(Op):
        # Properties attribute
        __props__ = ()

        #itypes and otypes attributes are
        #compulsory if make_node method is not defined.
        #They're the type of input and output respectively
        itypes = None
        otypes = None

        #Compulsory if itypes and otypes are not defined
        def make_node(self, *inputs):
            pass

        # Python implementation:
        def perform(self, node, inputs_storage, output_storage):
            pass

        # Other type of implementation
        # C implementation: [see aesara web site for other functions]
        def c_code(self, node, inputs, outputs, sub):
            pass

        # Other implementations:
        def make_thunk(self, node, storage_map, _, _2, impl=None):
            pass

        # optional:
        check_input = True

        def __init__(self, *args):
            pass

        def grad(self, inputs, g):
            pass

        def R_op(self, inputs, eval_points):
            pass

        def infer_shape(self, fgraph, node, input_shapes):
            pass

An :class:`Op` has to implement some methods defined in the the interface of
:class:`Op`. More specifically, it is mandatory for an :class:`Op` to define either
the method :meth:`Op.make_node` or :attr:`Op.itypes`, :attr:`Op.otypes` and one of the
implementation methods, either :meth:`Op.perform`, :meth:`COp.c_code`
or :meth:`Op.make_thunk`.

  :meth:`Op.make_node` method creates an Apply node representing the application
  of the :class:`Op` on the inputs provided. This method is responsible for three things:

    - it first checks that the input :class:`Variable`\s types are compatible
      with the current :class:`Op`. If the :class:`Op` cannot be applied on the provided
      input types, it must raises an exception (such as :class:`TypeError`).
    - it operates on the :class:`Variable`\s found in
      ``*inputs`` in Aesara's symbolic language to infer the type of
      the symbolic output :class:`Variable`\s. It creates output :class:`Variable`\s of a suitable
      symbolic :class:`Type` to serve as the outputs of this :class:`Op`'s
      application.
    - it creates an :class:`Apply` instance with the input and output :class:`Variable`, and
      return the :class:`Apply` instance.



  :meth:`Op.perform` method defines the Python implementation of an :class:`Op`.
  It takes several arguments:

    - ``node`` is a reference to an Apply node which was previously
      obtained via the :meth:`Op.make_node` method. It is typically not
      used in a simple :class:`Op`, but it contains symbolic information that
      could be required by a complex :class:`Op`.
    - ``inputs`` is a list of references to data which can be operated on using
      non-symbolic statements, (i.e., statements in Python, Numpy).
    - ``output_storage`` is a list of storage cells where the output
      is to be stored. There is one storage cell for each output of the :class:`Op`.
      The data put in ``output_storage`` must match the type of the
      symbolic output. It is forbidden to change the length of the list(s)
      contained in ``output_storage``.
      A function Mode may allow ``output_storage`` elements to persist
      between evaluations, or it may reset ``output_storage`` cells to
      hold a value of ``None``.  It can also pre-allocate some memory
      for the :class:`Op` to use.  This feature can allow ``perform`` to reuse
      memory between calls, for example. If there is something
      preallocated in the ``output_storage``, it will be of the good
      dtype, but can have the wrong shape and have any stride pattern.

  :meth:`Op.perform` method must be determined by the inputs. That is to say,
  when applied to identical inputs the method must return the same outputs.

  An :class:`Op`\s implementation can be defined in other ways, as well.
  For instance, it is possible to define a C-implementation via :meth:`COp.c_code`.
  Please refers to tutorial :ref:`creating_a_c_op` for a description of
  :meth:`COp.c_code` and other related ``c_**`` methods. Note that an
  :class:`Op` can provide both Python and C implementations.

  :meth:`Op.make_thunk` method is another alternative to :meth:`Op.perform`.
  It returns a thunk. A thunk is defined as a zero-arguments
  function which encapsulates the computation to be performed by an
  :class:`Op` on the arguments of its corresponding node. It takes several parameters:

    - ``node`` is the :class:`Apply` instance for which a thunk is requested,
    - ``storage_map`` is a ``dict`` of lists which  maps variables to a one-element
      lists holding the variable's current value. The one-element list acts as
      pointer to the value and allows sharing that "pointer" with other nodes
      and instances.
    - ``compute_map`` is also a  dict of lists.
      It maps variables to one-element lists holding booleans.  If
      the value is 0 then the variable has not been computed and the
      value should not be considered valid.  If the value is 1 the
      variable has been computed and the value is valid.  If the value
      is 2 the variable has been garbage-collected and is no longer
      valid, but shouldn't be required anymore for this call.
      The returned function must ensure that it sets the computed
      variables as computed in the :obj:`compute_map`.
    - ``impl`` allow to select between multiple implementation.
      It should have a default value of ``None``.

  :meth:`Op.make_thunk` is useful if you want to generate code and compile
  it yourself.

  If :meth:`Op.make_thunk` is defined by an :class:`Op`, it will be used by Aesara
  to obtain the :class:`Op`'s implementation.
  :meth:`Op.perform` and :meth:`COp.c_code` will be ignored.

  If :meth:`Op.make_node` is not defined, the :attr:`Op.itypes` and :attr:`Op.otypes`
  are used by the :class:`Op`'s :meth:`Op.make_node` method to implement the functionality
  of :meth:`Op.make_node` method mentioned above.

:class:`Op`'s auxiliary methods
-------------------------------

There are other methods that can be optionally defined by the :class:`Op`:

  :meth:`Op.__eq__` and :meth:`Op.__hash__` define respectivelly equality
  between two :class:`Op`\s and the hash of an :class:`Op` instance.
  They will be used during the rewriting phase to merge nodes that are doing
  equivalent computations (same inputs, same operation).
  Two :class:`Op`\s that are equal according :meth:`Op.__eq__`
  should return the same output when they are applied on the same inputs.

  The :attr:`Op.__props__` attribute lists the properties that influence how the computation
  is performed. Usually these are set in :meth:`Op.__init__`. It must be a tuple.
  If you don't have any properties, then you should set this attribute to the
  empty tuple ``()``.

  :attr:`Op.__props__` enables the  automatic generation of appropriate
  :meth:`Op.__eq__` and :meth:`Op.__hash__`.
  Given the method :func:`__eq__`, automatically generated from
  :attr:`Op.__props__`, two :class:`Op`\s will be equal if they have the same values for all
  the properties listed in :attr:`Op.__props__`.
  Given to the method :meth:`Op.__hash__` automatically generated from
  :attr:`Op.__props__`, two :class:`Op`\s will be have the same hash if they have the same
  values for all the properties listed in :attr:`Op.__props__`.
  :attr:`Op.__props__` will also generate a  suitable :meth:`Op.__str__` for your :class:`Op`.

  The :meth:`Op.infer_shape` method allows an :class:`Op` to infer the shape of its
  output variables without actually computing them.
  It takes as input ``fgraph``, a :class:`FunctionGraph`; ``node``, a reference
  to the :class:`Op`'s :class:`Apply` node;
  and a list of :class:`Variables`\s (e.g. ``i0_shape``, ``i1_shape``, ...)
  which are the dimensions of the :class:`Op` input :class:`Variable`\s.
  :meth:`Op.infer_shape` returns a list where each element is a tuple representing
  the shape of one output.
  This could be helpful if one only needs the shape of the output instead of the
  actual outputs, which can be useful, for instance, for rewriting
  procedures.

  The :meth:`Op.grad` method is required if you want to differentiate some cost
  whose expression includes your :class:`Op`. The gradient may be
  specified symbolically in this method. It takes two arguments ``inputs`` and
  ``output_gradients``, which are both lists of :class:`Variable`\s, and
  those must be operated on using Aesara's symbolic language. The :meth:`Op.grad`
  method must return a list containing one :class:`Variable` for each
  input. Each returned :class:`Variable` represents the gradient with respect
  to that input computed based on the symbolic gradients with respect
  to each output.
  If the output is not differentiable with respect to an input then
  this method should be defined to return a variable of type :class:`NullType`
  for that input. Likewise, if you have not implemented the gradient
  computation for some input, you may return a variable of type
  :class:`NullType` for that input. Please refer to :meth:`Op.grad` for a more detailed
  view.

  The :meth:`Op.R_op` method is needed if you want :func:`aesara.gradient.Rop` to
  work with your :class:`Op`.
  This function implements the application of the R-operator on the
  function represented by your :class:`Op`. Let assume that function is :math:`f`,
  with input :math:`x`, applying the R-operator means computing the
  Jacobian of :math:`f` and right-multiplying it by :math:`v`, the evaluation
  point, namely: :math:`\frac{\partial f}{\partial x} v`.

  The optional boolean :attr:`check_input` attribute is used to specify
  if you want the types used in your :class:`COp` to check their inputs in their
  :meth:`COp.c_code`. It can be used to speed up compilation, reduce overhead
  (particularly for scalars) and reduce the number of generated C files.


Example: :class:`Op` definition
-------------------------------

.. testcode:: example

    import aesara
    from aesara.graph.op import Op
    from aesara.graph.basic import Apply


    class DoubleOp1(Op):
        __props__ = ()

        def make_node(self, x):
            x = aesara.tensor.as_tensor_variable(x)
            # Note: using x_.type() is dangerous, as it copies x's broadcasting
            # behaviour
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, output_storage):
            x = inputs[0]
            z = output_storage[0]
            z[0] = x * 2

        def infer_shape(self, fgraph, node, i0_shapes):
            return i0_shapes

        def grad(self, inputs, output_grads):
            return [output_grads[0] * 2]

        def R_op(self, inputs, eval_points):
            # R_op can receive None as eval_points.
            # That mean there is no diferientiable path through that input
            # If this imply that you cannot compute some outputs,
            # return None for those.
            if eval_points[0] is None:
                return eval_points
            return self.grad(inputs, eval_points)

    doubleOp1 = DoubleOp1()

    #Using itypes and otypes


    class DoubleOp2(Op):
        __props__ = ()

        itypes = [aesara.tensor.dmatrix]
        otypes = [aesara.tensor.dmatrix]

        def perform(self, node, inputs, output_storage):
            x = inputs[0]
            z = output_storage[0]
            z[0] = x * 2

        def infer_shape(self, fgraph, node, i0_shapes):
            return i0_shapes

        def grad(self, inputs, output_grads):
            return [output_grads[0] * 2]

        def R_op(self, inputs, eval_points):
            # R_op can receive None as eval_points.
            # That mean there is no diferientiable path through that input
            # If this imply that you cannot compute some outputs,
            # return None for those.
            if eval_points[0] is None:
                return eval_points
            return self.grad(inputs, eval_points)

    doubleOp2 = DoubleOp2()

At a high level, the code fragment declares a class (e.g., ``DoubleOp1``) and then
creates one instance of it (e.g., ``doubleOp1``).

We often gloss over this distinction, but will be precise here:
``doubleOp1`` (the instance) is an :class:`Op`, not ``DoubleOp1`` (the class which is a
subclass of :class:`Op`). You can call ``doubleOp1(tensor.vector())`` on a
``Variable`` to build an expression, and in the expression there will be
a ``.op`` attribute that refers to ``doubleOp1``.

.. The first two methods in the :class:`Op` are relatively boilerplate: ``__eq__``
.. and ``__hash__``.
.. When two :class:`Op`\s are equal, Aesara will merge their outputs if they are applied to the same inputs.
.. The base class says two objects are equal if (and only if)
.. they are the same object.
.. Writing these boilerplate definitions ensures that the logic of the equality comparison is always explicit.

.. It is an essential part of the :ref:`op_contract` that if two :class:`Op`\s compare
.. equal, then they must compute the same result when presented with the same
.. inputs.  Here, if we allocated another instance of ``Fibby`` by typing ``fibby2
.. = Fibby()`` then we would have two :class:`Op`\s that behave identically.
..
.. When should the implementation of ``__eq__`` be more complicated?
.. If ``Fibby.__init__`` had parameters, then we could
.. have configured ``fibby2`` differently from ``fibby`` by passing different
.. arguments to the constructor. If we had done that, and if that different
.. configuration made ``fibby2`` compute different results from ``fibby`` (for the
.. same inputs) then we would have to add logic to the ``__eq__`` and ``__hash__``
.. function so that he two ``Fibby`` :class:`Op`\s would *not be equal*.  The reason why: Aesara's merge
.. optimization looks for :class:`Op`\s comparing equal and merges them. If two :class:`Op`\s compare
.. equal but don't always produce equal results from equal inputs, then you might
.. see wrong calculation.

The ``make_node`` method creates a node to be included in the expression graph.
It runs when we apply our :class:`Op` (``doubleOp1``) to the ``Variable`` (``x``), as
in ``doubleOp1(tensor.vector())``.
When an :class:`Op` has multiple inputs, their order in the inputs argument to ``Apply``
is important:  Aesara will call ``make_node(*inputs)`` to copy the graph,
so it is important not to change the semantics of the expression by changing
the argument order.

All the ``inputs`` and ``outputs`` arguments to :class:`Apply` must be :class:`Variable`\s.
A common and easy way to ensure inputs are variables is to run them through
``as_tensor_variable``. This function leaves :class:`TensorType` variables alone, raises
an error for non-:class:`TensorType` variables, and copies any ``numpy.ndarray`` into
the storage for a :class:`TensorType` :class:`Constant`. The :func:`make_node` method dictates the
appropriate :class:`Type` for all output variables.

The :func:`perform` method implements the :class:`Op`'s mathematical logic in Python.
The inputs (here ``x``) are passed by value, but a single output is returned
indirectly as the first element of single-element lists.  If ``doubleOp1`` had
a second output, it would be stored in ``output_storage[1][0]``.

In some execution modes, the output storage might contain the return value of
a previous call.  That old value can be reused to avoid memory re-allocation,
but it must not influence the semantics of the :class:`Op` output.

You can try the new :class:`Op` as follows:

.. testcode:: example

    import numpy as np
    import aesara

    x = aesara.tensor.matrix()
    f = aesara.function([x], DoubleOp1()(x))
    inp = np.random.random_sample((5, 4))
    out = f(inp)
    assert np.allclose(inp * 2, out)
    print(inp)
    print(out)

.. testoutput:: example
   :hide:
   :options: +ELLIPSIS, +SKIP

    <BLANKLINE>

.. code-block:: none

    [[ 0.08257206  0.34308357  0.5288043   0.06582951]
     [ 0.65977826  0.10040307  0.5402353   0.55472296]
     [ 0.82358552  0.29502171  0.97387481  0.0080757 ]
     [ 0.77327215  0.65401857  0.76562992  0.94145702]
     [ 0.8452076   0.30500101  0.88430501  0.95818655]]
    [[ 0.16514411  0.68616713  1.0576086   0.13165902]
     [ 1.31955651  0.20080613  1.08047061  1.10944593]
     [ 1.64717104  0.59004341  1.94774962  0.0161514 ]
     [ 1.5465443   1.30803715  1.53125983  1.88291403]
     [ 1.6904152   0.61000201  1.76861002  1.9163731 ]]

.. testcode:: example

    import numpy as np
    import aesara

    x = aesara.tensor.matrix()
    f = aesara.function([x], DoubleOp2()(x))
    inp = np.random.random_sample((5, 4))
    out = f(inp)
    assert np.allclose(inp * 2, out)
    print(inp)
    print(out)


.. testoutput:: example
   :hide:
   :options: +ELLIPSIS, +SKIP

    <BLANKLINE>

.. code-block:: none

    [[ 0.02443785  0.67833979  0.91954769  0.95444365]
     [ 0.60853382  0.7770539   0.78163219  0.92838837]
     [ 0.04427765  0.37895602  0.23155797  0.4934699 ]
     [ 0.20551517  0.7419955   0.34500905  0.49347629]
     [ 0.24082769  0.49321452  0.24566545  0.15351132]]
    [[ 0.04887571  1.35667957  1.83909538  1.90888731]
     [ 1.21706764  1.55410779  1.56326439  1.85677674]
     [ 0.08855531  0.75791203  0.46311594  0.9869398 ]
     [ 0.41103034  1.48399101  0.69001811  0.98695258]
     [ 0.48165539  0.98642904  0.4913309   0.30702264]]


Example: :attr:`__props__` definition
-------------------------------------

We can modify the previous piece of code in order to demonstrate
the usage of the :attr:`__props__` attribute.

We create an :class:`Op` that takes a variable ``x`` and returns ``a*x+b``.
We want to say that two such :class:`Op`\s are equal when their values of ``a``
and ``b`` are equal.

.. testcode:: properties

    import aesara
    from aesara.graph.op import Op
    from aesara.graph.basic import Apply


    class AXPBOp(Op):
        """
        This creates an Op that takes x to a*x+b.
        """
        __props__ = ("a", "b")

        def __init__(self, a, b):
            self.a = a
            self.b = b
            super().__init__()

        def make_node(self, x):
            x = aesara.tensor.as_tensor_variable(x)
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, output_storage):
            x = inputs[0]
            z = output_storage[0]
            z[0] = self.a * x + self.b

        def infer_shape(self, fgraph, node, i0_shapes):
            return i0_shapes

        def grad(self, inputs, output_grads):
            return [self.a * output_grads[0]]


The use of :attr:`__props__` saves
the user the trouble of implementing :func:`__eq__` and :func:`__hash__`
manually. It also generates a default :func:`__str__` method that prints the
attribute names and their values.

We can test this by running the following segment:

.. testcode:: properties

    mult4plus5op = AXPBOp(4, 5)
    another_mult4plus5op = AXPBOp(4, 5)
    mult2plus3op = AXPBOp(2, 3)

    assert mult4plus5op == another_mult4plus5op
    assert mult4plus5op != mult2plus3op

    x = aesara.tensor.matrix()
    f = aesara.function([x], mult4plus5op(x))
    g = aesara.function([x], mult2plus3op(x))

    inp = np.random.random_sample((5, 4)).astype(np.float32)
    assert np.allclose(4 * inp + 5, f(inp))
    assert np.allclose(2 * inp + 3, g(inp))


How To Test it
--------------

Aesara has some functionalities to simplify testing. These help test the
:meth:`Op.infer_shape`, :meth:`Op.grad` and :meth:`Op.R_op` methods. Put the following code
in a file and execute it with the ``pytest`` program.

Basic Tests
^^^^^^^^^^^

Basic tests are done by you just by using the :class:`Op` and checking that it
returns the right answer. If you detect an error, you must raise an
exception. You can use the ``assert`` keyword to automatically raise an
`AssertionError`.

.. testcode:: tests

    import numpy as np
    import aesara
    from tests import unittest_tools as utt


    class TestDouble(utt.InferShapeTester):
        def setup_method(self):
            super().setup_method()
            self.op_class = DoubleOp
            self.op = DoubleOp()

        def test_basic(self):
            rng = np.random.default_rng(utt.fetch_seed())

            x = aesara.tensor.matrix()
            f = aesara.function([x], self.op(x))

            inp = np.asarray(rng.random((5, 4)), dtype=aesara.config.floatX)
            out = f(inp)
            # Compare the result computed to the expected value.
            utt.assert_allclose(inp * 2, out)

We call ``utt.assert_allclose(expected_value, value)`` to compare
NumPy ndarray.This raise an error message with more information. Also,
the default tolerance can be changed with the Aesara flags
``config.tensor__cmp_sloppy`` that take values in 0, 1 and 2. The
default value do the most strict comparison, 1 and 2 make less strict
comparison.

Testing the :meth:`Op.infer_shape`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a class inherits from the :class:`InferShapeTester` class, it gets the
:meth:`InferShapeTester._compile_and_check` method that tests the :meth:`Op.infer_shape`
method. It tests that the :class:`Op` gets rewritten out of the graph if only
the shape of the output is needed and not the output
itself. Additionally, it checks that the rewritten graph computes
the correct shape, by comparing it to the actual shape of the computed
output.

:meth:`InferShapeTester._compile_and_check` compiles an Aesara function. It takes as
parameters the lists of input and output Aesara variables, as would be
provided to :func:`aesara.function`, and a list of real values to pass to the
compiled function. It also takes the :class:`Op` class as a parameter
in order to verify that no instance of it appears in the shape-optimized graph.

If there is an error, the function raises an exception. If you want to
see it fail, you can implement an incorrect :meth:`Op.infer_shape`.

When testing with input values with shapes that take the same value
over different dimensions (for instance, a square matrix, or a ``tensor3``
with shape ``(n, n, n)``, or ``(m, n, m)``), it is not possible to detect if
the output shape was computed correctly, or if some shapes with the
same value have been mixed up. For instance, if the :meth:`Op.infer_shape` uses
the width of a matrix instead of its height, then testing with only
square matrices will not detect the problem. This is why the
:meth:`InferShapeTester._compile_and_check` method prints a warning in such a case. If
your :class:`Op` works only with such matrices, you can disable the warning with the
``warn=False`` parameter.

.. testcode:: tests

    from aesara.configdefaults import config
    from tests import unittest_tools as utt


    class TestDouble(utt.InferShapeTester):

        # [...] as previous tests.

        def test_infer_shape(self):
            rng = np.random.default_rng(utt.fetch_seed())
            x = aesara.tensor.matrix()
            self._compile_and_check(
                [x],  # aesara.function inputs
                [self.op(x)],  # aesara.function outputs
                # Always use not square matrix!
                # inputs data
                [np.asarray(rng.random((5, 4)), dtype=config.floatX)],
                # Op that should be removed from the graph.
                self.op_class,
            )

Testing the gradient
^^^^^^^^^^^^^^^^^^^^

The function :ref:`verify_grad <validating_grad>`
verifies the gradient of an :class:`Op` or Aesara graph. It compares the
analytic (symbolically computed) gradient and the numeric
gradient (computed through the Finite Difference Method).

If there is an error, the function raises an exception. If you want to
see it fail, you can implement an incorrect gradient (for instance, by removing
the multiplication by 2).

.. testcode:: tests

        def test_grad(self):
            rng = np.random.default_rng(utt.fetch_seed())
            tests.unittest_tools.verify_grad(
                self.op,
                [rng.random((5, 7, 2))]
            )

Testing the Rop
^^^^^^^^^^^^^^^

.. TODO: repair defective links in the following paragraph

The class :class:`RopLop_checker` defines the functions
:func:`RopLop_checker.check_mat_rop_lop`, :func:`RopLop_checker.check_rop_lop` and
:func:`RopLop_checker.check_nondiff_rop`. These allow to test the
implementation of the :meth:`Rop` method of a particular :class:`Op`.

For instance, to verify the :meth:`Rop` method of the ``DoubleOp``, you can use this:

.. testcode:: tests

   import numpy
   import tests
   from tests.test_rop import RopLop_checker
   class TestDoubleRop(RopLop_checker):
       def setUp(self):
           super(TestDoubleRop, self).setUp()
       def test_double_rop(self):
           self.check_rop_lop(DoubleRop()(self.x), self.in_shape)

Running Your Tests
^^^^^^^^^^^^^^^^^^

To perform your tests, simply run ``pytest``.

In-file
"""""""

One may also add a block of code similar to the following at the end
of the file containing a specific test of interest and run the
file. In this example, the test ``TestDoubleRop`` in the class
``test_double_op`` would be performed.

.. testcode:: tests

    if __name__ == '__main__':
       t = TestDoubleRop("test_double_rop")
       t.setUp()
       t.test_double_rop()

We recommend that when we execute a file, we run all tests in that
file. This can be done by adding this at the end of your test files:

.. testcode:: tests

    if __name__ == '__main__':
        unittest.main()

Exercise
""""""""

Run the code of the ``DoubleOp`` example above.

Modify and execute to compute: ``x * y``.

Modify and execute the example to return two outputs: ``x + y`` and `jx - yj`.

You can omit the :meth:`Rop` functions. Try to implement the testing apparatus
described above.

(Notice that Aesara's current *elemwise fusion* rewrite is
only applicable to computations involving a single output. Hence, to gain
efficiency over the basic solution that is asked here, the two operations would
have to be jointly rewritten explicitly in the code.)

Random numbers in tests
"""""""""""""""""""""""

Making tests errors more reproducible is a good practice. To make
tests more reproducible, one needs a way to get the same random
numbers. This can be done by seeding NumPy's random number
generator.

For convenience, the classes :class:`InferShapeTester` and :class:`RopLop_checker`
already do this for you. If you implement your own :meth:`setUp` method,
don't forget to call the parent :meth:`setUp` method.


:download:`Solution<extending_aesara_solution_1.py>`


:func:`as_op`
---------------------

:func:`as_op` is a Python decorator that converts a Python function into a
basic Aesara :class:`Op` that will call the supplied function during execution.

This isn't the recommended way to build an :class:`Op`, but allows for a quick
implementation.

It takes an optional :meth:`Op.infer_shape` parameter that must have this
signature:

.. code-block:: none

    def infer_shape(fgraph, node, input_shapes):
        # ...
        return output_shapes

  - :obj:`input_shapes` and :obj:`output_shapes` are lists of tuples that
    represent the shape of the corresponding inputs/outputs, and :obj:`fgraph`
    is a :class:`FunctionGraph`.

.. warning::

    Not providing a :meth:`Op.infer_shape` prevents shape-related
    rewrites from working with this :class:`Op`. For example
    ``your_op(inputs, ...).shape`` will need the :class:`Op` to be executed just
    to get the shape.

.. note::

    As no grad is defined, this means you won't be able to
    differentiate paths that include this :class:`Op`.

.. note::

    It converts the Python function to a callable object that takes as
    inputs Aesara variables that were declared.

.. note::
    The python function wrapped by the :func:`as_op` decorator needs to return a new
    data allocation, no views or in place modification of the input.

:func:`as_op` Example
^^^^^^^^^^^^^^^^^^^^^

.. testcode:: asop

    import aesara
    import aesara.tensor as at
    import numpy as np
    from aesara import function
    from aesara.compile.ops import as_op


    def infer_shape_numpy_dot(fgraph, node, input_shapes):
        ashp, bshp = input_shapes
        return [ashp[:-1] + bshp[-1:]]


    @as_op(itypes=[at.matrix, at.matrix],
           otypes=[at.matrix], infer_shape=infer_shape_numpy_dot)
    def numpy_dot(a, b):
       return np.dot(a, b)

You can try it as follows:

.. testcode:: asop

    x = at.matrix()
    y = at.matrix()
    f = function([x, y], numpy_dot(x, y))
    inp1 = np.random.random_sample((5, 4))
    inp2 = np.random.random_sample((4, 7))
    out = f(inp1, inp2)


.. _Documentation:

Documentation and Coding Style
------------------------------
Please always respect the :ref:`quality_contributions` or your contribution
will not be accepted.

:class:`NanGuardMode` and :class:`AllocEmpty`
---------------------------------------------

:class:`NanGuardMode` help users find where in the graph ``NaN`` appear. But
sometimes, we want some variables to not be checked. For example, in
the old GPU back-end, we used a float32 :class:`CudaNdarray` to store the MRG
random number generator state (they are integers). So if :class:`NanGuardMode`
checked it, it would generate a false positive. Another case is related to
:class:`AllocEmpty` or some computations on it (like done by :class:`Scan`).

You can tell :class:`NanGuardMode` to do not check a variable with:
:attr:`variable.tag.nan_guard_mode_check`. Also, this tag automatically
follows that variable during rewriting. This mean if you tag a
variable that get replaced by an inplace version, it will keep that
tag.


Final Note
----------

A more extensive discussion of this section's content may be found in
the advanced tutorial :ref:`Extending Aesara<extending>`.

The section :ref:`Other Ops <other_ops>` includes more instructions for
the following specific cases:

 - :ref:`scalar_ops`
 - :ref:`sparse_ops`
 - :ref:`Random ops <random_ops>`
 - :ref:`openmp_ops`
 - :ref:`numba_ops`

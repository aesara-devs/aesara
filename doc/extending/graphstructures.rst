
.. _graphstructures:

================
Graph Structures
================

Aesara works by modeling mathematical operations and their outputs using
symbolic placeholders, or :term:`variables <variable>`, which inherit from the class
:class:`Variable`.  When writing expressions in Aesara one uses operations like
``+``, ``-``, ``**``, ``sum()``, ``tanh()``. These are represented
internally as :term:`Op`\s.  An :class:`Op` represents a computation that is
performed on a set of symbolic inputs and produces a set of symbolic outputs.
These symbolic input and output :class:`Variable`\s carry information about
their types, like their data type (e.g. float, int), the number of dimensions,
etc.

Aesara graphs are composed of interconnected :term:`Apply`, :term:`Variable` and
:class:`Op` nodes. An :class:`Apply` node represents the application of an
:class:`Op` to specific :class:`Variable`\s. It is important to draw the
difference between the definition of a computation represented by an :class:`Op`
and its application to specific inputs, which is represented by the
:class:`Apply` node.

The following illustrates these elements:

**Code**

.. testcode::

   import aesara.tensor as at

   x = at.dmatrix('x')
   y = at.dmatrix('y')
   z = x + y

**Diagram**

.. _tutorial-graphfigure:

.. image:: apply.png
    :align: center


The blue box is an :class:`Apply` node. Red boxes are :class:`Variable`\s. Green
circles are :class:`Op`\s. Purple boxes are :class:`Type`\s.

.. TODO
    Clarify the 'acyclic' graph and the 'back' pointers or references that
    'don't count'.

When we create :class:`Variable`\s and then :class:`Apply`
:class:`Op`\s to them to make more :class:`Variable`\s, we build a
bi-partite, directed, acyclic graph. :class:`Variable`\s point to the :class:`Apply` nodes
representing the function application producing them via their
:attr:`Variable.owner` field. These :class:`Apply` nodes point in turn to their input and
output :class:`Variable`\s via their :attr:`Apply.inputs` and :attr:`Apply.outputs` fields.

The :attr:`Variable.owner` field of both ``x`` and ``y`` point to ``None`` because
they are not the result of another computation. If one of them was the
result of another computation, its :attr:`Variable.owner` field would point to another
blue box like ``z`` does, and so on.


Traversing the graph
====================

The graph can be traversed starting from outputs (the result of some
computation) down to its inputs using the owner field.
Take for example the following code:

>>> import aesara
>>> x = aesara.tensor.dmatrix('x')
>>> y = x * 2.

If you enter ``type(y.owner)`` you get ``<class 'aesara.graph.basic.Apply'>``,
which is the :class:`Apply` node that connects the :class:`Op` and the inputs to get this
output. You can now print the name of the :class:`Op` that is applied to get
``y``:

>>> y.owner.op.name
'Elemwise{mul,no_inplace}'

Hence, an element-wise multiplication is used to compute ``y``. This
multiplication is done between the inputs:

>>> len(y.owner.inputs)
2
>>> y.owner.inputs[0]
x
>>> y.owner.inputs[1]
InplaceDimShuffle{x,x}.0

Note that the second input is not ``2`` as we would have expected. This is
because ``2`` was first :term:`broadcasted <broadcasting>` to a matrix of
same shape as ``x``. This is done by using the :class:`Op`\ :class:`DimShuffle`:

>>> type(y.owner.inputs[1])
<class 'aesara.tensor.var.TensorVariable'>
>>> type(y.owner.inputs[1].owner)
<class 'aesara.graph.basic.Apply'>
>>> y.owner.inputs[1].owner.op # doctest: +SKIP
<aesara.tensor.elemwise.DimShuffle object at 0x106fcaf10>
>>> y.owner.inputs[1].owner.inputs
[TensorConstant{2.0}]

All of the above can be succinctly summarized with the :func:`aesara.dprint`
function:

>>> aesara.dprint(y)
Elemwise{mul,no_inplace} [id A] ''
 |x [id B]
 |InplaceDimShuffle{x,x} [id C] ''
   |TensorConstant{2.0} [id D]

Starting from this graph structure it is easier to understand how
*automatic differentiation* proceeds and how the symbolic relations
can be rewritten for performance or stability.


Graph Structures
================

The following section outlines each type of structure that may be used
in an Aesara-built computation graph.


.. index::
   single: Apply
   single: graph construct; Apply

.. _apply:

:class:`Apply`
--------------

An :class:`Apply` node is a type of internal node used to represent a
:term:`computation graph <graph>` in Aesara. Unlike
:class:`Variable`, :class:`Apply` nodes are usually not
manipulated directly by the end user. They may be accessed via
the :attr:`Variable.owner` field.

An :class:`Apply` node is typically an instance of the :class:`Apply`
class. It represents the application
of an :class:`Op` on one or more inputs, where each input is a
:class:`Variable`. By convention, each :class:`Op` is responsible for
knowing how to build an :class:`Apply` node from a list of
inputs. Therefore, an :class:`Apply` node may be obtained from an :class:`Op`
and a list of inputs by calling ``Op.make_node(*inputs)``.

Comparing with the Python language, an :class:`Apply` node is
Aesara's version of a function call whereas an :class:`Op` is
Aesara's version of a function definition.

An :class:`Apply` instance has three important fields:

**op**
  An :class:`Op` that determines the function/transformation being
  applied here.

**inputs**
  A list of :class:`Variable`\s that represent the arguments of
  the function.

**outputs**
  A list of :class:`Variable`\s that represent the return values
  of the function.

An :class:`Apply` instance can be created by calling ``graph.basic.Apply(op, inputs, outputs)``.



.. index::
   single: Op
   single: graph construct; Op

.. _op:


:class:`Op`
-----------

An :class:`Op` in Aesara defines a certain computation on some types of
inputs, producing some types of outputs. It is equivalent to a
function definition in most programming languages. From a list of
input :ref:`Variables <variable>` and an :class:`Op`, you can build an :ref:`apply`
node representing the application of the :class:`Op` to the inputs.

It is important to understand the distinction between an :class:`Op` (the
definition of a function) and an :class:`Apply` node (the application of a
function). If you were to interpret the Python language using Aesara's
structures, code going like ``def f(x): ...`` would produce an :class:`Op` for
``f`` whereas code like ``a = f(x)`` or ``g(f(4), 5)`` would produce an
:class:`Apply` node involving the ``f`` :class:`Op`.


.. index::
   single: Type
   single: graph construct; Type

.. _type:

:class:`Type`
-------------

A :class:`Type` in Aesara provides static information (or constraints) about
data objects in a graph. The information provided by :class:`Type`\s allows
Aesara to perform rewrites and produce more efficient compiled code.

Every symbolic :class:`Variable` in an Aesara graph has an associated
:class:`Type` instance, and :class:`Type`\s also serve as a means of
constructing :class:`Variable` instances.  In other words, :class:`Type`\s and
:class:`Variable`\s go hand-in-hand.

For example, :ref:`aesara.tensor.irow <libdoc_tensor_creation>` is an instance of a
:class:`Type` and it can be used to construct variables as follows:

>>> from aesara.tensor import irow
>>> irow()
<TensorType(int32, (1, ?))>

As the string print-out shows, `irow` specifies the following information about
the :class:`Variable`\s it constructs:

#. They represent tensors that are backed by :class:`numpy.ndarray`\s.
   This comes from the fact that `irow` is an instance of :class:`TensorType`,
   which is the base :class:`Type` for symbolic :class:`numpy.ndarray`\s.
#. They represent arrays of 32-bit integers (i.e. from the ``int32``).
#. They represent arrays with shapes of :math:`1 \times N`, or, in code, ``(1,
   None)``, where ``None`` represents any shape value.

Note that Aesara :class:`Type`\s are not necessarily equivalent to Python types or
classes. Aesara's :class:`TensorType`'s, like `irow`, use :class:`numpy.ndarray`
as the underlying Python type for performing computations and storing data, but
:class:`numpy.ndarray`\s model a much wider class of arrays than most :class:`TensorType`\s.
In other words, Aesara :class:`Type`'s try to be more specific.

For more information see :ref:`aesara_type`.

.. index::
   single: Variable
   single: graph construct; Variable

.. _variable:

:class:`Variable`
-----------------

A :class:`Variable` is the main data structure you work with when using
Aesara. The symbolic inputs that you operate on are :class:`Variable`\s and what
you get from applying various :class:`Op`\s to these inputs are also
:class:`Variable`\s. For example, when one inputs

>>> import aesara
>>> x = aesara.tensor.ivector()
>>> y = -x

``x`` and ``y`` are both :class:`Variable`\s. The :class:`Type` of both ``x`` and
``y`` is `aesara.tensor.ivector`.

Unlike ``x``, ``y`` is a :class:`Variable` produced by a computation (in this
case, it is the negation of ``x``). ``y`` is the :class:`Variable` corresponding to
the output of the computation, while ``x`` is the :class:`Variable`
corresponding to its input. The computation itself is represented by
another type of node, an :class:`Apply` node, and may be accessed
through ``y.owner``.

More specifically, a :class:`Variable` is a basic structure in Aesara that
represents a datum at a certain point in computation. It is typically
an instance of the class :class:`Variable` or
one of its subclasses.

A :class:`Variable` ``r`` contains four important fields:

**type**
  a :class:`Type` defining the kind of value this :class:`Variable` can hold in
  computation.

**owner**
  this is either ``None`` or an :class:`Apply` node of which the :class:`Variable` is
  an output.

**index**
  the integer such that ``owner.outputs[index] is r`` (ignored if
  :attr:`Variable.owner` is ``None``)

**name**
  a string to use in pretty-printing and debugging.

:class:`Variable` has an important subclass: :ref:`Constant <constant>`.

.. index::
   single: Constant
   single: graph construct; Constant

.. _constant:


:class:`Constant`
^^^^^^^^^^^^^^^^^

A :class:`Constant` is a :class:`Variable` with one extra, immutable field:
:attr:`Constant.data`.
When used in a computation graph as the input of an
:class:`Op`\ :class:`Apply`, it is assumed that said input
will *always* take the value contained in the :class:`Constant`'s data
field. Furthermore, it is assumed that the :class:`Op` will not under
any circumstances modify the input. This means that a :class:`Constant` is
eligible to participate in numerous rewrites: constant in-lining
in C code, constant folding, etc.

Automatic Differentiation
=========================

Having the graph structure, computing automatic differentiation is
simple. The only thing :func:`aesara.grad` has to do is to traverse the
graph from the outputs back towards the inputs through all :class:`Apply`
nodes. For each such :class:`Apply` node, its :class:`Op` defines
how to compute the gradient of the node's outputs with respect to its
inputs. Note that if an :class:`Op` does not provide this information,
it is assumed that the gradient is not defined.

Using the `chain rule <http://en.wikipedia.org/wiki/Chain_rule>`_,
these gradients can be composed in order to obtain the expression of the
gradient of the graph's output with respect to the graph's inputs.

A following section of this tutorial will examine the topic of
:ref:`differentiation<tutcomputinggrads>` in greater detail.

Rewrites
========

When compiling an Aesara graph using :func:`aesara.function`, a graph is
necessarily provided.  While this graph structure shows how to compute the
output from the input, it also offers the possibility to improve the way this
computation is carried out. The way rewrites work in Aesara is by
identifying and replacing certain patterns in the graph with other specialized
patterns that produce the same results but are either faster or more
stable. Rewrites can also detect identical subgraphs and ensure that the
same values are not computed twice.

For example, one simple rewrite that Aesara uses is to replace
the pattern :math:`\frac{xy}{y}` by :math:`x`.

See :ref:`graph_rewriting` and :ref:`optimizations` for more information.

**Example**

Consider the following example of rewrites:

>>> import aesara
>>> a = aesara.tensor.vector("a")      # declare symbolic variable
>>> b = a + a ** 10                    # build symbolic expression
>>> f = aesara.function([a], b)        # compile function
>>> print(f([0, 1, 2]))                # prints `array([0,2,1026])`
[    0.     2.  1026.]
>>> aesara.printing.pydotprint(b, outfile="./pics/symbolic_graph_no_rewrite.png", var_with_name_simple=True)  # doctest: +SKIP
The output file is available at ./pics/symbolic_graph_no_rewrite.png
>>> aesara.printing.pydotprint(f, outfile="./pics/symbolic_graph_rewite.png", var_with_name_simple=True)  # doctest: +SKIP
The output file is available at ./pics/symbolic_graph_rewrite.png

We used :func:`aesara.printing.pydotprint` to visualize the rewritten graph
(right), which is much more compact than the un-rewritten graph (left).

.. |g1| image:: ./pics/symbolic_graph_unopt.png
        :width: 500 px
.. |g2| image:: ./pics/symbolic_graph_opt.png
        :width: 500 px

================================ ====================== ================================
        Un-rewritten graph                                      Rewritten graph
================================ ====================== ================================
|g1|                                                              |g2|
================================ ====================== ================================

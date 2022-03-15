
.. _graph_rewriting:

===============
Graph Rewriting
===============

In this document we will explain how graph rewriting works and how graph
optimizations can be constructed using graph rewriting.

.. todo::
   The old "optimization" nomenclature is still in use throughout these documents
   and the codebase; however, this is being changed to more accurately distinguish
   between general graph rewriting for any purpose and the kind that is explicitly
   intended to "optimize" a graph in some way.


Global and Local Optimizations
==============================

First, let's lay out the way optimizations work in Aesara. There are
two types of optimizations: *global* optimizations and *local*
optimizations. A global optimization takes a :class:`FunctionGraph` object (see its
:doc:`documentation </library/graph/fgraph>` for more details) and navigates through it
in a suitable way, replacing some :class:`Variable`\s by others in the process. A
local optimization, on the other hand, is defined as a function on a
*single* :ref:`apply` node and must return either ``False`` (to mean that
nothing is to be done) or a list of new :class:`Variable`\s that we would like to
replace the node's outputs with. A :ref:`navigator` is a special kind
of global optimization which navigates the computation graph in some
fashion (e.g. in topological order, reverse-topological order, random
order, etc.) and applies one or more local optimizations at each step.

Optimizations which are holistic, meaning that they must take into
account dependencies that might be all over the graph, should be
global. Optimizations that can be done with a narrow perspective are
better defined as local optimizations. The majority of optimizations
we want to define are local.

.. optimizer:

Global optimization
-------------------

.. class:: GlobalOptimizer

    .. method:: apply(fgraph)

      This method takes a :class:`FunctionGraph` object which contains the computation graph
      and does modifications in line with what the optimization is meant
      to do. This is one of the main methods of the optimizer.

    .. method:: add_requirements(fgraph)

      This method takes a :class:`FunctionGraph` object and adds :ref:`features
      <libdoc_graph_fgraphfeature>` to it. These features are "plugins" that are needed
      for the :meth:`GlobalOptimizer.apply` method to do its job properly.

    .. method:: optimize(fgraph)

      This is the interface function called by Aesara.  It calls
      :meth:`GlobalOptimizer.apply` by default.


Local optimization
------------------

A local optimization is an object which defines the following methods:

.. class:: LocalOptimizer

    .. method:: transform(fgraph, node)

      This method takes a :class:`FunctionGraph` and an :class:`Apply` node and
      returns either ``False`` to signify that no changes are to be done or a
      list of :class:`Variable`\s which matches the length of the node's ``outputs``
      list. When the :class:`LocalOptimizer` is applied by a :class:`NavigatorOptimizer`, the outputs
      of the node passed as argument to the :class:`LocalOptimizer` will be replaced by
      the list returned.


A simplification rule
=====================

For starters, let's define the following simplification:

.. math::

   \frac{xy}{y} = x

We will implement it in three ways: using a global optimization, a
local optimization with a :class:`NavigatorOptimizer` and then using the :class:`PatternSub`
facility.

Global optimization
-------------------

Here is the code for a global optimization implementing the
simplification described above:

.. testcode::

   import aesara
   from aesara.graph.opt import GlobalOptimizer
   from aesara.graph.features import ReplaceValidate

   class Simplify(GlobalOptimizer):
       def add_requirements(self, fgraph):
           fgraph.attach_feature(ReplaceValidate())

       def apply(self, fgraph):
           for node in fgraph.toposort():
               if node.op == true_div:
                   x, y = node.inputs
                   z = node.outputs[0]
                   if x.owner and x.owner.op == mul:
                       a, b = x.owner.inputs
                       if y == a:
                           fgraph.replace_validate(z, b)
                       elif y == b:
                           fgraph.replace_validate(z, a)

   simplify = Simplify()


Here's how it works: first, in :meth:`add_requirements`, we add the
:class:`ReplaceValidate` :class:`Feature` located in
:ref:`libdoc_graph_features`. This feature adds the :meth:`replace_validate`
method to ``fgraph``, which is an enhanced version of :meth:`FunctionGraph.replace` that
does additional checks to ensure that we are not messing up the
computation graph.

In a nutshell, :class:`ReplaceValidate` grants access to :meth:`fgraph.replace_validate`,
and :meth:`fgraph.replace_validate` allows us to replace a :class:`Variable` with
another while respecting certain validation constraints. As an
exercise, try to rewrite :class:`Simplify` using :class:`NodeFinder`. (Hint: you
want to use the method it publishes instead of the call to toposort)

Then, in :meth:`GlobalOptimizer.apply` we do the actual job of simplification. We start by
iterating through the graph in topological order. For each node
encountered, we check if it's a ``div`` node. If not, we have nothing
to do here. If so, we put in ``x``, ``y`` and ``z`` the numerator,
denominator and quotient (output) of the division.
The simplification only occurs when the numerator is a multiplication,
so we check for that. If the numerator is a multiplication we put the
two operands in ``a`` and ``b``, so
we can now say that ``z == (a*b)/y``. If ``y==a`` then ``z==b`` and if
``y==b`` then ``z==a``. When either case happens then we can replace
``z`` by either ``a`` or ``b`` using :meth:`FunctionGraph.replace_validate`; otherwise, we do
nothing.

Now, we test the optimization:

>>> from aesara.scalar import float64, add, mul, true_div
>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = add(z, mul(true_div(mul(y, x), y), true_div(z, x)))
>>> e = aesara.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(add(z, mul(true_div(mul(y, x), y), true_div(z, x))))
>>> simplify.optimize(e)
>>> e
FunctionGraph(add(z, mul(x, true_div(z, x))))

You can check what happens if you put many
instances of :math:`\frac{xy}{y}` in the graph. Note that it sometimes
won't work for reasons that have nothing to do with the quality of the
optimization you wrote. For example, consider the following:

>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = true_div(mul(add(y, z), x), add(y, z))
>>> e = aesara.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(true_div(mul(add(y, z), x), add(y, z)))
>>> simplify.optimize(e)
>>> e
FunctionGraph(true_div(mul(add(y, z), x), add(y, z)))

Nothing happened here. The reason is: ``add(y, z) != add(y,
z)``. That is the case for efficiency reasons. To fix this problem we
first need to merge the parts of the graph that represent the same
computation, using the :class:`MergeOptimizer` defined in
:mod:`aesara.graph.opt`.

>>> from aesara.graph.opt import MergeOptimizer
>>> MergeOptimizer().optimize(e)  # doctest: +ELLIPSIS
(0, ..., None, None, {}, 1, 0)
>>> e
FunctionGraph(true_div(mul(*1 -> add(y, z), x), *1))
>>> simplify.optimize(e)
>>> e
FunctionGraph(x)

Once the merge is done, both occurrences of ``add(y, z)`` are
collapsed into a single one and is used as an input in two
places. Note that ``add(x, y)`` and ``add(y, x)`` are still considered
to be different because Aesara has no clue that ``add`` is
commutative. You may write your own global optimizer to identify
computations that are identical with full knowledge of the rules of
arithmetic that your Ops implement. Aesara might provide facilities
for this somewhere in the future.

.. note::

   :class:`FunctionGraph` is an Aesara structure intended for the optimization
   phase. It is used internally by :func:`aesara.function` and is rarely
   exposed to the end user.


Local Optimization
------------------

The local version of the above code would be the following:


.. testcode::

   from aesara.graph.opt import LocalOptimizer


   class LocalSimplify(LocalOptimizer):
       def transform(self, fgraph, node):
           if node.op == true_div:
               x, y = node.inputs
               if x.owner and x.owner.op == mul:
                   a, b = x.owner.inputs
                   if y == a:
                       return [b]
                   elif y == b:
                       return [a]
           return False

       def tracks(self):
           # This tells certain navigators to only apply this `LocalOptimizer`
           # on these kinds of `Op`s
           return [true_div]

   local_simplify = LocalSimplify()


In this case, the transformation is defined in the
:meth:`LocalOptimizer.transform` method, which is given an explicit
:class:`Apply` node on which to work.  The entire graph--as a ``fgraph``--is
also provided, in case global information is needed.

If no changes are to be made, ``False`` must be returned; otherwise, a list of replacements for the node's
outputs are returned. This list must have the same length as
:attr:`node.outputs`. If one of :attr:`node.outputs` doesn't have clients
(e.g. available via ``fgraph.clients``), then it is not used elsewhere in the graph and
you can put ``None`` in the returned list to remove it.

In order to apply the local optimizer we can use it in conjunction
with a :class:`NavigatorOptimizer`. Basically, a :class:`NavigatorOptimizer` is
a global optimizer that loops through all nodes in the graph (or a well-defined
subset of them) and applies one or several local optimizers.

>>> x = float64('x')
>>> y = float64('y')
>>> z = float64('z')
>>> a = add(z, mul(true_div(mul(y, x), y), true_div(z, x)))
>>> e = aesara.graph.fg.FunctionGraph([x, y, z], [a])
>>> e
FunctionGraph(add(z, mul(true_div(mul(y, x), y), true_div(z, x))))
>>> simplify = aesara.graph.opt.TopoOptimizer(local_simplify)
>>> simplify.optimize(e)
(<aesara.graph.opt.TopoOptimizer object at 0x...>, 1, 5, 3, ..., ..., ...)
>>> e
FunctionGraph(add(z, mul(x, true_div(z, x))))

:class:`OpSub`, :class:`OpRemove`, :class:`PatternSub`
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Aesara defines some shortcuts to make :class:`LocalOptimizer`\s:

.. function:: OpSub(op1, op2)

  Replaces all uses of ``op1`` by ``op2``. In other
  words, the outputs of all :class:`Apply` nodes using ``op1`` by the outputs
  of :class:`Apply` nodes involving ``op2``, where their inputs are the same.

.. function:: OpRemove(op)

  Removes all uses of ``op`` in the following way:
  if ``y = op(x)`` then ``y`` is replaced by ``x``. ``op`` must have as many
  outputs as it has inputs. The first output becomes the first input,
  the second output becomes the second input, and so on.

.. function:: PatternSub(pattern1, pattern2)

  Replaces all occurrences of the first pattern by the second pattern.
  See :class:`PatternSub`.

.. code::

   from aesara.scalar import identity
   from aesara.graph.opt import OpSub, OpRemove, PatternSub

   # Replacing `add` by `mul` (this is not recommended for primarily
   # mathematical reasons):
   add_to_mul = OpSub(add, mul)

   # Removing `identity`
   remove_identity = OpRemove(identity)

   # The "simplify" operation we've been defining in the past few
   # sections. Note that we need two patterns to account for the
   # permutations of the arguments to `mul`.
   local_simplify_1 = PatternSub((true_div, (mul, 'x', 'y'), 'y'), 'x')
   local_simplify_2 = PatternSub((true_div, (mul, 'x', 'y'), 'x'), 'y')

.. note::

   :class:`OpSub`, :class:`OpRemove` and :class:`PatternSub` produce local optimizers, which
   means that everything we said previously about local optimizers
   apply (e.g. they need to be wrapped in a :class:`NavigatorOptimizer`, etc.)


When an optimization can be naturally expressed using :class:`OpSub`, :class:`OpRemove`
or :class:`PatternSub`, it is highly recommended to use them.

.. _unification:

Unification and reification
===========================

The :class:`PatternSub` class uses `unification and reification
<https://en.wikipedia.org/wiki/Unification_(computer_science)>`_ to implement a
more succinct and reusable form of "pattern matching and replacement".
In general, *use of the unification and reification tools is preferable when
a rewrite's matching and replacement are non-trivial*, so we will briefly explain
them in the following.

Aesara's unification and reification tools are provided by the
`logical-unification <https://github.com/pythological/unification>`_ package.
The basic tools are :func:`unify`, :func:`reify`, and :class:`var`.  The class :class:`var`
construct *logic variables*, which represent the elements to be unified/matched, :func:`unify`
performs the "matching", and :func:`reify` performs the "replacements".

See :mod:`unification`'s documentation for an introduction to unification and reification.

In order to use :func:`unify` and :func:`reify` with Aesara graphs, we need an intermediate
structure that will allow us to represent Aesara graphs that contain :class:`var`\s, because
Aesara :class:`Op`\s and :class:`Apply` nodes will not accept these foreign objects as inputs.

:class:`PatternSub` uses Python ``tuple``\s to effectively represent :class:`Apply` nodes and
``str``\s to represent logic variables (i.e. :class:`var`\s in the :mod:`unification` library).
Behind the scenes, these ``tuple``\s are converted to a ``tuple`` subclass called :class:`ExpressionTuple`\s,
which behave just like normal ``tuple``\s except for some special caching features that allow for easy
evaluation and caching.  These :class:`ExpressionTuple`\s are provided by the
`etuples <https://github.com/pythological/etuples>`_ library.

Here is an illustration of all the above components used together:

>>> from unification import unify, reify, var
>>> from etuples import etuple
>>> y_lv = var()  # Create a logic variable
>>> y_lv
~_1
>>> s = unify(add(x, y), etuple(add, x, y_lv))
>>> s
{~_1: y}

In this example, :func:`unify` matched the Aesara graph in the first argument with the "pattern"
given by the :func:`etuple` in the second.  The result is a ``dict`` mapping logic variables to
the objects to which they were successfully unified.  When a :func:`unify` doesn't succeed, it will
return ``False``.

:func:`reify` uses ``dict``\s like the kind produced by :func:`unify` to replace
logic variables within structures:

>>> res = reify(etuple(add, y_lv, y_lv), s)
>>> res
e(<aesara.scalar.basic.Add at 0x7f54dfa5a350>, y, y)

Since :class:`ExpressionTuple`\s can be evaluated, we can produce a complete Aesara graph from these
results as follows:

>>> res.evaled_obj
add.0
>>> aesara.dprint(res.evaled_obj)
add [id A] ''
 |y [id B]
 |y [id B]


Because :class:`ExpressionTuple`\s effectively model `S-expressions
<https://en.wikipedia.org/wiki/S-expression>`_, they can be used with the `cons
<https://github.com/pythological/python-cons>`_ package to unify and reify
graphs structurally.

Let's say we want to match graphs that use the :class:`add`\ :class:`Op` but could have a
varying number of arguments:

>>> from cons import cons
>>> op_lv = var()
>>> args_lv = var()
>>> s = unify(cons(op_lv, args_lv), add(x, y))
>>> s
{~_2: <aesara.scalar.basic.Add at 0x7f54dfa5a350>, ~_3: e(x, y)}
>>> s = unify(cons(op_lv, args_lv), add(x, y, z))
>>> s
{~_2: <aesara.scalar.basic.Add at 0x7f54dfa5a350>, ~_3: e(x, y, z)}

From here, we can check ``s[op_lv] == add`` to confirm that we have the correct :class:`Op` and
proceed with our rewrite.

>>> res = reify(cons(mul, args_lv), s)
>>> res
e(<aesara.scalar.basic.Mul at 0x7f54dfa5ae10>, x, y, z)
>>> aesara.dprint(res.evaled_obj)
mul [id A] ''
 |x [id B]
 |y [id C]
 |z [id D]


.. _miniKanren_rewrites:

miniKanren
==========

Given that unification and reification are fully implemented for Aesara objects via the :mod:`unificiation` package,
the `kanren <https://github.com/pythological/kanren>`_ package can be used with Aesara graphs, as well.
:mod:`kanren` implements the `miniKanren <http://minikanren.org/>`_ domain-specific language for relational programming.

Refer to the links above for a proper introduction to miniKanren, but suffice it to say that
miniKanren orchestrates the unification and reification operations described in :ref:`unification`, and
it does so in the context of relational operators (e.g. equations like :math:`x + x = 2 x`).
This means that a relation that--say--represents :math:`x + x = 2 x` can be
utilized in both directions.

Currently, the local optimizer :class:`KanrenRelationSub` provides a means of
turning :mod:`kanren` relations into :class:`LocalOptimizer`\s; however,
:mod:`kanren` can always be used directly from within a custom :class:`Rewriter`, so
:class:`KanrenRelationSub` is not necessary.

The following is an example that distributes dot products across additions.

.. code::

    import aesara
    import aesara.tensor as at
    from aesara.graph.kanren import KanrenRelationSub
    from aesara.graph.opt import EquilibriumOptimizer
    from aesara.graph.opt_utils import optimize_graph
    from aesara.tensor.math import _dot
    from etuples import etuple
    from kanren import conso, eq, fact, heado, tailo
    from kanren.assoccomm import assoc_flatten, associative
    from kanren.core import lall
    from kanren.graph import mapo
    from unification import vars as lvars


    # Make the graph pretty printing results a little more readable
    aesara.pprint.assign(
        _dot, aesara.printing.OperatorPrinter("@", -1, "left")
    )

    # Tell `kanren` that `add` is associative
    fact(associative, at.add)


    def dot_distributeo(in_lv, out_lv):
        """A `kanren` goal constructor relation for the relation ``A.dot(a + b ...) == A.dot(a) + A.dot(b) ...``."""
        A_lv, add_term_lv, add_cdr_lv, dot_cdr_lv, add_flat_lv = lvars(5)

        return lall(
            # Make sure the input is a `_dot`
            eq(in_lv, etuple(_dot, A_lv, add_term_lv)),
            # Make sure the term being `_dot`ed is an `add`
            heado(at.add, add_term_lv),
            # Flatten the associative pairings of `add` operations
            assoc_flatten(add_term_lv, add_flat_lv),
            # Get the flattened `add` arguments
            tailo(add_cdr_lv, add_flat_lv),
            # Add all the `_dot`ed arguments and set the output
            conso(at.add, dot_cdr_lv, out_lv),
            # Apply the `_dot` to all the flattened `add` arguments
            mapo(lambda x, y: conso(_dot, etuple(A_lv, x), y), add_cdr_lv, dot_cdr_lv),
        )


    dot_distribute_opt = EquilibriumOptimizer([KanrenRelationSub(dot_distributeo)], max_use_ratio=10)


Below, we apply `dot_distribute_opt` to a few example graphs.  First we create simple test graph:

>>> x_at = at.vector("x")
>>> y_at = at.vector("y")
>>> A_at = at.matrix("A")
>>> test_at = A_at.dot(x_at + y_at)
>>> print(aesara.pprint(test_at))
(A @ (x + y))

Next we apply the rewrite to the graph:

>>> res = optimize_graph(test_at, include=[], custom_opt=dot_distribute_opt, clone=False)
>>> print(aesara.pprint(res))
((A @ x) + (A @ y))

We see that the dot product has been distributed, as desired.  Now, let's try a
few more test cases:

>>> z_at = at.vector("z")
>>> w_at = at.vector("w")
>>> test_at = A_at.dot((x_at + y_at) + (z_at + w_at))
>>> print(aesara.pprint(test_at))
(A @ ((x + y) + (z + w)))
>>> res = optimize_graph(test_at, include=[], custom_opt=dot_distribute_opt, clone=False)
>>> print(aesara.pprint(res))
(((A @ x) + (A @ y)) + ((A @ z) + (A @ w)))

>>> B_at = at.matrix("B")
>>> w_at = at.vector("w")
>>> test_at = A_at.dot(x_at + (y_at + B_at.dot(z_at + w_at)))
>>> print(aesara.pprint(test_at))
(A @ (x + (y + ((B @ z) + (B @ w)))))
>>> res = optimize_graph(test_at, include=[], custom_opt=dot_distribute_opt, clone=False)
>>> print(aesara.pprint(res))
((A @ x) + ((A @ y) + ((A @ (B @ z)) + (A @ (B @ w)))))


This example demonstrates how non-trivial matching and replacement logic can
be neatly expressed in miniKanren's DSL, but it doesn't quite demonstrate miniKanren's
relational properties.

To do that, we will create another :class:`Rewriter` that simply reverses the arguments
to the relation :func:`dot_distributeo` and apply it to the distributed result in ``res``:

>>> dot_gather_opt = EquilibriumOptimizer([KanrenRelationSub(lambda x, y: dot_distributeo(y, x))], max_use_ratio=10)
>>> rev_res = optimize_graph(res, include=[], custom_opt=dot_gather_opt, clone=False)
>>> print(aesara.pprint(rev_res))
(A @ (x + (y + (B @ (z + w)))))

As we can see, the :mod:`kanren` relation works both ways, just like the underlying
mathematical relation does.

miniKanren relations can be used to explore rewrites of graphs in sophisticated
ways.  It also provides a framework that more directly maps to the mathematical
identities that drive graph rewrites.  For some simple examples of relational graph rewriting
in :mod:`kanren` see `here <https://github.com/pythological/kanren/blob/master/doc/graphs.md>`_.  For a
high-level overview of miniKanren's use as a tool for symbolic computation see
`"miniKanren as a Tool for Symbolic Computation in Python" <https://arxiv.org/abs/2005.11644>`_.


.. _optdb:

The optimization database (:obj:`optdb`)
========================================

Aesara exports a symbol called :obj:`optdb` which acts as a sort of
ordered database of optimizations. When you make a new optimization,
you must insert it at the proper place in the database. Furthermore,
you can give each optimization in the database a set of tags that can
serve as a basis for filtering.

The point of :obj:`optdb` is that you might want to apply many optimizations
to a computation graph in many unique patterns. For example, you might
want to do optimization X, then optimization Y, then optimization Z. And then
maybe optimization Y is an :class:`EquilibriumOptimizer` containing :class:`LocalOptimizer`\s A, B
and C which are applied on every node of the graph until they all fail to change
it. If some optimizations act up, we want an easy way to turn them off. Ditto if
some optimizations are very CPU-intensive and we don't want to take the time to
apply them.

The :obj:`optdb` system allows us to tag each optimization with a unique name
as well as informative tags such as 'stable', 'buggy' or
'cpu_intensive', all this without compromising the structure of the
optimizations.

For instance, the optimization tag ``cxx_only`` is used for optimizations that
insert :class:`Op`\s that have no Python implementation (i.e. they only have C
implementations).  Optimizations with this tag can be skipped when the C backend
is not being used.


Definition of :obj:`optdb`
--------------------------

:obj:`optdb` is an object which is an instance of
:class:`SequenceDB <optdb.SequenceDB>`,
itself a subclass of :class:`OptimizationDatabase <optdb.OptimizationDatabase>`.
There exist (for now) two types of :class:`OptimizationDatabase`, :class:`SequenceDB` and :class:`EquilibriumDB`.
When given an appropriate :class:`OptimizationQuery`, :class:`OptimizationDatabase` objects build an :class:`Optimizer` matching
the query.

A :class:`SequenceDB` contains :class:`Optimizer` or :class:`OptimizationDatabase` objects. Each of them
has a name, an arbitrary number of tags and an integer representing their order
in the sequence. When a :class:`OptimizationQuery` is applied to a :class:`SequenceDB`, all :class:`Optimizer`\s whose
tags match the query are inserted in proper order in a :class:`SequenceOptimizer`, which
is returned. If the :class:`SequenceDB` contains :class:`OptimizationDatabase`
instances, the :class:`OptimizationQuery` will be passed to them as well and the
optimizers they return will be put in their places.

An :class:`EquilibriumDB` contains :class:`LocalOptimizer` or :class:`OptimizationDatabase` objects. Each of them
has a name and an arbitrary number of tags. When a :class:`OptimizationQuery` is applied to
an :class:`EquilibriumDB`, all :class:`LocalOptimizer`\s that match the query are
inserted into an :class:`EquilibriumOptimizer`, which is returned. If the
:class:`SequenceDB` contains :class:`OptimizationDatabase` instances, the
:class:`OptimizationQuery` will be passed to them as well and the
:class:`LocalOptimizer`\s they return will be put in their places
(note that as of yet no :class:`OptimizationDatabase` can produce :class:`LocalOptimizer` objects, so this
is a moot point).

Aesara contains one principal :class:`OptimizationDatabase` object, :class:`optdb`, which
contains all of Aesara's optimizers with proper tags. It is
recommended to insert new :class:`Optimizer`\s in it. As mentioned previously,
optdb is a :class:`SequenceDB`, so, at the top level, Aesara applies a sequence
of global optimizations to the computation graphs.


:class:`OptimizationQuery`
--------------------------

A :class:`OptimizationQuery` is built by the following call:

.. code-block:: python

   aesara.graph.optdb.OptimizationQuery(include, require=None, exclude=None, subquery=None)

.. class:: OptimizationQuery

    .. attribute:: include

       A set of tags (a tag being a string) such that every
       optimization obtained through this :class:`OptimizationQuery` must have **one** of the tags
       listed. This field is required and basically acts as a starting point
       for the search.

    .. attribute:: require

       A set of tags such that every optimization obtained
       through this :class:`OptimizationQuery` must have **all** of these tags.

    .. attribute:: exclude

       A set of tags such that every optimization obtained
       through this :class:`OptimizationQuery` must have **none** of these tags.

    .. attribute:: subquery

       :obj:`optdb` can contain sub-databases; subquery is a
       dictionary mapping the name of a sub-database to a special :class:`OptimizationQuery`.
       If no subquery is given for a sub-database, the original :class:`OptimizationQuery` will be
       used again.

Furthermore, a :class:`OptimizationQuery` object includes three methods, :meth:`including`,
:meth:`requiring` and :meth:`excluding`, which each produce a new :class:`OptimizationQuery` object
with the include, require, and exclude sets refined to contain the new entries.


Examples
--------

Here are a few examples of how to use a :class:`OptimizationQuery` on :obj:`optdb` to produce an
:class:`Optimizer`:

.. testcode::

   from aesara.graph.optdb import OptimizationQuery
   from aesara.compile import optdb

   # This is how the optimizer for the fast_run mode is defined
   fast_run = optdb.query(OptimizationQuery(include=['fast_run']))

   # This is how the optimizer for the fast_compile mode is defined
   fast_compile = optdb.query(OptimizationQuery(include=['fast_compile']))

   # This is the same as fast_run but no optimizations will replace
   # any operation by an inplace version. This assumes, of course,
   # that all inplace operations are tagged as 'inplace' (as they
   # should!)
   fast_run_no_inplace = optdb.query(OptimizationQuery(include=['fast_run'],
                                           exclude=['inplace']))


Registering an :class:`Optimizer`
---------------------------------

Let's say we have a global optimizer called ``simplify``. We can add
it to :obj:`optdb` as follows:

.. testcode::

   # optdb.register(name, optimizer, order, *tags)
   optdb.register('simplify', simplify, 'fast_run', position=0.5)

Once this is done, the ``FAST_RUN`` mode will automatically include your
optimization (since you gave it the ``'fast_run'`` tag). Of course,
already-compiled functions will see no change. The 'order' parameter
(what it means and how to choose it) will be explained in
:ref:`optdb-structure` below.



Registering a :class:`LocalOptimizer`
-------------------------------------

:class:`LocalOptimizer`\s may be registered in two ways:

* Wrap them in a :class:`NavigatorOptimizer` and insert them like a global optimizer
  (see previous section).
* Put them in an :class:`EquilibriumDB`.

Aesara defines two :class:`EquilibriumDB`\s in which one can put local
optimizations:


.. function:: canonicalize

  This contains optimizations that aim to *simplify* the graph:

  * Replace rare or esoterical operations with their equivalents using
    elementary operations.

  * Order operations in a canonical way (any sequence of
    multiplications and divisions can be rewritten to contain at most
    one division, for example; ``x*x`` can be rewritten ``x**2``; etc.)

  * Fold constants (``Constant(2)*Constant(2)`` becomes ``Constant(4)``)


.. function:: specialize

  This contains optimizations that aim to *specialize* the graph:

  * Replace a combination of operations with a special operation that
    does the same thing (but better).


For each group, all optimizations of the group that are selected by
the :class:`OptimizationQuery` will be applied on the graph over and over again until none
of them is applicable, so keep that in mind when designing it: check
carefully that your optimization leads to a fixpoint (a point where it
cannot apply anymore) at which point it returns ``False`` to indicate its
job is done. Also be careful not to undo the work of another local
optimizer in the group, because then the graph will oscillate between
two or more states and nothing will get done.


.. _optdb-structure:

:obj:`optdb` structure
----------------------

:obj:`optdb` contains the following :class:`Optimizer`\s and sub-DBs, with the given
priorities and tags:

+-------+---------------------+------------------------------+
| Order | Name                | Description                  |
+=======+=====================+==============================+
| 0     | merge1              | First merge operation        |
+-------+---------------------+------------------------------+
| 1     | canonicalize        | Simplify the graph           |
+-------+---------------------+------------------------------+
| 2     | specialize          | Add specialized operations   |
+-------+---------------------+------------------------------+
| 49    | merge2              | Second merge operation       |
+-------+---------------------+------------------------------+
| 49.5  | add_destroy_handler | Enable inplace optimizations |
+-------+---------------------+------------------------------+
| 100   | merge3              | Third merge operation        |
+-------+---------------------+------------------------------+

The merge operations are meant to put together parts of the graph that
represent the same computation. Since optimizations can modify the
graph in such a way that two previously different-looking parts of the
graph become similar, we merge at the beginning, in the middle and at
the very end. Technically, we only really need to do it at the end,
but doing it in previous steps reduces the size of the graph and
therefore increases the efficiency of the process.

See previous section for more information about the canonicalize and
specialize steps.

The ``add_destroy_handler`` step is not really an optimization. It is
a marker. Basically:

.. warning::

   Any optimization which inserts inplace operations in the
   computation graph must appear **after** the ``add_destroy_handler``
   "optimizer". In other words, the priority of any such optimization
   must be **>= 50**. Failure to comply by this restriction can lead
   to the creation of incorrect computation graphs.

The reason the destroy handler is not inserted at the beginning is
that it is costly to run. It is cheaper to run most optimizations
under the assumption there are no inplace operations.


.. _navigator:

:class:`NavigatorOptimizer`
---------------------------

WRITEME

.. _profiling_opt:

Profiling Aesara function compilation
=====================================

You find that compiling an Aesara function is taking too much time? You
can get profiling information about Aesara optimization. The normal
:ref:`Aesara profiler <tut_profiling>` will provide you with very
high-level information. The indentation shows the included in/subset
relationship between sections. The top of its output look like this:

.. code-block:: none

    Function profiling
    ==================
      Message: PATH_TO_A_FILE:23
      Time in 0 calls to Function.__call__: 0.000000e+00s
      Total compile time: 1.131874e+01s
        Number of Apply nodes: 50
        Aesara Optimizer time: 1.152431e+00s
           Aesara validate time: 2.790451e-02s
        Aesara Linker time (includes C, CUDA code generation/compiling): 7.893991e-02s
           Import time 1.153541e-02s
      Time in all call to aesara.grad() 4.732513e-02s

Explanations:

* ``Total compile time: 1.131874e+01s`` gives the total time spent inside `aesara.function`.
* ``Number of Apply nodes: 50`` means that after optimization, there are 50 apply node in the graph.
* ``Aesara Optimizer time: 1.152431e+00s`` means that we spend 1.15s in the ``aesara.function`` phase where we optimize (modify) the graph to make it faster / more stable numerically /...
* ``Aesara validate time: 2.790451e-02s`` means that we spent 2.8e-2s in the *validate* subset of the optimization phase.
* ``Aesara Linker time (includes C code generation/compiling): 7.893991e-02s`` means that we spent 7.9e-2s in *linker* phase of ``aesara.function``.
* ``Import time 1.153541e-02s`` is a subset of the linker time where we import the compiled module.
* ``Time in all call to aesara.grad() 4.732513e-02s`` tells that we spent a total of 4.7e-2s in all calls to ``aesara.grad``. This is outside of the calls to ``aesara.function``.

The *linker* phase includes the generation of the C code, the time spent
by g++ to compile and the time needed by Aesara to build the object we
return. The C code generation and compilation is cached, so the first
time you compile a function and the following ones could take different
amount of execution time.

Detailed profiling of Aesara optimizations
------------------------------------------

You can get more detailed profiling information about the Aesara
optimizer phase by setting to ``True`` the Aesara flags
:attr:`config.profile_optimizer` (this requires ``config.profile`` to be ``True``
as well).

This will output something like this:

.. code-block:: none

    Optimizer Profile
    -----------------
     SeqOptimizer  OPT_FAST_RUN  time 1.152s for 123/50 nodes before/after optimization
       0.028s for fgraph.validate()
       0.131s for callback
       time      - (name, class, index) - validate time
       0.751816s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.004s
         EquilibriumOptimizer      canonicalize
           time 0.751s for 14 passes
           nb nodes (start, end,  max) 108 81 117
           time io_toposort 0.029s
           time in local optimizers 0.687s
           time in global optimizers 0.010s
            0 - 0.050s 27 (0.000s in global opts, 0.002s io_toposort) - 108 nodes - ('local_dimshuffle_lift', 9) ('local_upcast_elemwise_constant_inputs', 5) ('local_shape_to_shape_i', 3) ('local_fill_sink', 3) ('local_fill_to_alloc', 2) ...
            1 - 0.288s 26 (0.002s in global opts, 0.002s io_toposort) - 117 nodes - ('local_dimshuffle_lift', 8) ('local_fill_sink', 4) ('constant_folding', 4) ('local_useless_elemwise', 3) ('local_subtensor_make_vector', 3) ...
            2 - 0.044s 13 (0.002s in global opts, 0.003s io_toposort) - 96 nodes - ('constant_folding', 4) ('local_dimshuffle_lift', 3) ('local_fill_sink', 3) ('local_useless_elemwise', 1) ('local_fill_to_alloc', 1) ...
            3 - 0.045s 11 (0.000s in global opts, 0.002s io_toposort) - 91 nodes - ('constant_folding', 3) ('local_fill_to_alloc', 2) ('local_dimshuffle_lift', 2) ('local_mul_canonizer', 2) ('MergeOptimizer', 1) ...
            4 - 0.035s 8 (0.002s in global opts, 0.002s io_toposort) - 93 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
            5 - 0.035s 6 (0.000s in global opts, 0.002s io_toposort) - 88 nodes - ('local_fill_sink', 2) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('local_mul_canonizer', 1)
            6 - 0.038s 10 (0.001s in global opts, 0.002s io_toposort) - 95 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 3) ('constant_folding', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1)
            7 - 0.032s 5 (0.001s in global opts, 0.002s io_toposort) - 91 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1)
            8 - 0.034s 5 (0.000s in global opts, 0.002s io_toposort) - 92 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_greedy_distributor', 1)
            9 - 0.031s 6 (0.001s in global opts, 0.002s io_toposort) - 90 nodes - ('local_fill_sink', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1) ('local_greedy_distributor', 1)
           10 - 0.032s 5 (0.000s in global opts, 0.002s io_toposort) - 89 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_fill_sink', 1)
           11 - 0.030s 5 (0.000s in global opts, 0.002s io_toposort) - 88 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
           12 - 0.026s 1 (0.000s in global opts, 0.003s io_toposort) - 81 nodes - ('MergeOptimizer', 1)
           13 - 0.031s 0 (0.000s in global opts, 0.003s io_toposort) - 81 nodes -
           times - times applied - nb node created - name:
           0.263s - 15 - 0 - constant_folding
           0.096s - 2 - 14 - local_greedy_distributor
           0.066s - 4 - 19 - local_mul_canonizer
           0.046s - 28 - 57 - local_fill_sink
           0.042s - 35 - 78 - local_dimshuffle_lift
           0.018s - 5 - 15 - local_upcast_elemwise_constant_inputs
           0.010s - 11 - 4 - MergeOptimizer
           0.009s - 4 - 0 - local_useless_elemwise
           0.005s - 11 - 2 - local_fill_to_alloc
           0.004s - 3 - 6 - local_neg_to_mul
           0.002s - 1 - 3 - local_lift_transpose_through_dot
           0.002s - 3 - 4 - local_shape_to_shape_i
           0.002s - 2 - 4 - local_subtensor_lift
           0.001s - 3 - 0 - local_subtensor_make_vector
           0.001s - 1 - 1 - local_sum_all_to_none
           0.131s - in 62 optimization that where not used (display only those with a runtime > 0)
             0.050s - local_add_canonizer
             0.018s - local_mul_zero
             0.016s - local_one_minus_erf
             0.010s - local_func_inv
             0.006s - local_0_dot_x
             0.005s - local_track_shape_i
             0.004s - local_mul_switch_sink
             0.004s - local_fill_cut
             0.004s - local_one_minus_erf2
             0.003s - local_remove_switch_const_cond
             0.003s - local_cast_cast
             0.002s - local_IncSubtensor_serialize
             0.001s - local_sum_div_dimshuffle
             0.001s - local_div_switch_sink
             0.001s - local_dimshuffle_no_inplace_at_canonicalize
             0.001s - local_cut_useless_reduce
             0.001s - local_reduce_join
             0.000s - local_sum_sum
             0.000s - local_useless_alloc
             0.000s - local_reshape_chain
             0.000s - local_useless_subtensor
             0.000s - local_reshape_lift
             0.000s - local_flatten_lift
             0.000s - local_useless_slice
             0.000s - local_subtensor_of_alloc
             0.000s - local_subtensor_of_dot
             0.000s - local_subtensor_merge
       0.101733s - ('elemwise_fusion', 'SeqOptimizer', 13) - 0.000s
         SeqOptimizer      elemwise_fusion  time 0.102s for 78/50 nodes before/after optimization
           0.000s for fgraph.validate()
           0.004s for callback
           0.095307s - ('composite_elemwise_fusion', 'FusionOptimizer', 1) - 0.000s
             FusionOptimizer
              nb_iter 3
              nb_replacement 10
              nb_inconsistency_replace 0
              validate_time 0.000249624252319
              callback_time 0.00316381454468
              time_toposort 0.00375390052795
           0.006412s - ('local_add_mul_fusion', 'FusionOptimizer', 0) - 0.000s
             FusionOptimizer
              nb_iter 2
              nb_replacement 3
              nb_inconsistency_replace 0
              validate_time 6.43730163574e-05
              callback_time 0.000783205032349
              time_toposort 0.0035240650177
       0.090089s - ('inplace_elemwise_optimizer', 'FromFunctionOptimizer', 30) - 0.019s
       0.048993s - ('BlasOpt', 'SeqOptimizer', 8) - 0.000s
         SeqOptimizer      BlasOpt  time 0.049s for 81/80 nodes before/after optimization
           0.000s for fgraph.validate()
           0.003s for callback
           0.035997s - ('gemm_optimizer', 'GemmOptimizer', 1) - 0.000s
             GemmOptimizer
              nb_iter 2
              nb_replacement 2
              nb_replacement_didn_t_remove 0
              nb_inconsistency_make 0
              nb_inconsistency_replace 0
              time_canonicalize 0.00720071792603
              time_factor_can 9.05990600586e-06
              time_factor_list 0.00128507614136
              time_toposort 0.00311398506165
              validate_time 4.60147857666e-05
              callback_time 0.00174236297607
           0.004569s - ('local_dot_to_dot22', 'TopoOptimizer', 0) - 0.000s
             TopoOptimizer
               nb_node (start, end, changed) (81, 81, 5)
               init io_toposort 0.00139284133911
               loop time 0.00312399864197
               callback_time 0.00172805786133
           0.002283s - ('local_dot22_to_dot22scalar', 'TopoOptimizer', 2) - 0.000s
             TopoOptimizer
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.00171804428101
               loop time 0.000502109527588
               callback_time 0.0
           0.002257s - ('local_gemm_to_gemv', 'EquilibriumOptimizer', 3) - 0.000s
             EquilibriumOptimizer          local_gemm_to_gemv
               time 0.002s for 1 passes
               nb nodes (start, end,  max) 80 80 80
               time io_toposort 0.001s
               time in local optimizers 0.000s
               time in global optimizers 0.000s
                0 - 0.002s 0 (0.000s in global opts, 0.001s io_toposort) - 80 nodes -
           0.002227s - ('use_c_blas', 'TopoOptimizer', 4) - 0.000s
             TopoOptimizer
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.0014750957489
               loop time 0.00068998336792
               callback_time 0.0
           0.001632s - ('use_scipy_ger', 'TopoOptimizer', 5) - 0.000s
             TopoOptimizer
               nb_node (start, end, changed) (80, 80, 0)
               init io_toposort 0.00138401985168
               loop time 0.000202178955078
               callback_time 0.0
       0.031740s - ('specialize', 'EquilibriumOptimizer', 9) - 0.000s
         EquilibriumOptimizer      specialize
           time 0.031s for 2 passes
           nb nodes (start, end,  max) 80 78 80
           time io_toposort 0.003s
           time in local optimizers 0.022s
           time in global optimizers 0.004s
            0 - 0.017s 6 (0.002s in global opts, 0.001s io_toposort) - 80 nodes - ('constant_folding', 2) ('local_mul_to_sqr', 1) ('local_elemwise_alloc', 1) ('local_div_to_inv', 1) ('local_mul_specialize', 1)
            1 - 0.014s 0 (0.002s in global opts, 0.001s io_toposort) - 78 nodes -
           times - times applied - nb node created - name:
           0.003s - 1 - 1 - local_mul_specialize
           0.002s - 1 - 2 - local_elemwise_alloc
           0.002s - 2 - 0 - constant_folding
           0.001s - 1 - 1 - local_div_to_inv
           0.001s - 1 - 1 - local_mul_to_sqr
           0.016s - in 69 optimization that where not used (display only those with a runtime > 0)
             0.004s - crossentropy_to_crossentropy_with_softmax_with_bias
             0.002s - local_one_minus_erf
             0.002s - Elemwise{sub,no_inplace}(z, Elemwise{mul,no_inplace}(alpha subject to <function <lambda> at 0x7f475e4da050>, SparseDot(x, y))) -> Usmm{no_inplace}(Elemwise{neg,no_inplace}(alpha), x, y, z)
             0.002s - local_add_specialize
             0.001s - local_func_inv
             0.001s - local_useless_elemwise
             0.001s - local_abs_merge
             0.001s - local_track_shape_i
             0.000s - local_one_minus_erf2
             0.000s - local_sum_mul_by_scalar
             0.000s - local_elemwise_sub_zeros
             0.000s - local_cast_cast
             0.000s - local_alloc_unary
             0.000s - Elemwise{log,no_inplace}(Softmax(x)) -> <function make_out_pattern at 0x7f47619a8410>(x)
             0.000s - local_sum_div_dimshuffle
             0.000s - local_sum_alloc
             0.000s - local_dimshuffle_lift
             0.000s - local_reduce_broadcastable
             0.000s - local_grad_log_erfc_neg
             0.000s - local_advanced_indexing_crossentropy_onehot
             0.000s - local_log_erfc
             0.000s - local_log1p
             0.000s - local_log_add
             0.000s - local_useless_alloc
             0.000s - local_neg_neg
             0.000s - local_neg_div_neg
    ...


To understand this profile here is some explanation of how optimizations work:

* Optimizations are organized in an hierarchy. At the top level, there
  is a :class:`SeqOptimizer`. It contains other optimizers,
  and applies them in the order they were specified. Those sub-optimizers can be
  of other types, but are all *global* optimizers.

* Each :class:`Optimizer` in the hierarchy will print some stats about
  itself. The information that it prints depends of the type of the
  optimizer.

* The :class:`SeqOptimizer` will print some stats at the start:

    .. code-block:: none

        Optimizer Profile
        -----------------
         SeqOptimizer  OPT_FAST_RUN  time 1.152s for 123/50 nodes before/after optimization
           0.028s for fgraph.validate()
           0.131s for callback
           time      - (name, class, index) - validate time

  Then it will print, with some additional indentation, each sub-optimizer's profile
  information. These sub-profiles are ordered by the time they took to execute,
  not by their execution order.

  * ``OPT_FAST_RUN`` is the name of the optimizer
  * 1.152s is the total time spent in that optimizer
  * 123/50 means that before this optimization, there were 123 apply node in the function graph, and after only 50.
  * 0.028s means it spent that time calls to ``fgraph.validate()``
  * 0.131s means it spent that time for callbacks. This is a mechanism that can trigger other execution when there is a change to the FunctionGraph.
  * ``time      - (name, class, index) - validate time`` tells how the information for each sub-optimizer get printed.
  * All other instances of :class:`SeqOptimizer` are described like this. In
    particular, some sub-optimizer from ``OPT_FAST_RUN`` that are also
    :class:`SeqOptimizer`.


* The :class:`SeqOptimizer` will print some stats at the start:

    .. code-block:: none

       0.751816s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.004s
         EquilibriumOptimizer      canonicalize
           time 0.751s for 14 passes
           nb nodes (start, end,  max) 108 81 117
           time io_toposort 0.029s
           time in local optimizers 0.687s
           time in global optimizers 0.010s
            0 - 0.050s 27 (0.000s in global opts, 0.002s io_toposort) - 108 nodes - ('local_dimshuffle_lift', 9) ('local_upcast_elemwise_constant_inputs', 5) ('local_shape_to_shape_i', 3) ('local_fill_sink', 3) ('local_fill_to_alloc', 2) ...
            1 - 0.288s 26 (0.002s in global opts, 0.002s io_toposort) - 117 nodes - ('local_dimshuffle_lift', 8) ('local_fill_sink', 4) ('constant_folding', 4) ('local_useless_elemwise', 3) ('local_subtensor_make_vector', 3) ...
            2 - 0.044s 13 (0.002s in global opts, 0.003s io_toposort) - 96 nodes - ('constant_folding', 4) ('local_dimshuffle_lift', 3) ('local_fill_sink', 3) ('local_useless_elemwise', 1) ('local_fill_to_alloc', 1) ...
            3 - 0.045s 11 (0.000s in global opts, 0.002s io_toposort) - 91 nodes - ('constant_folding', 3) ('local_fill_to_alloc', 2) ('local_dimshuffle_lift', 2) ('local_mul_canonizer', 2) ('MergeOptimizer', 1) ...
            4 - 0.035s 8 (0.002s in global opts, 0.002s io_toposort) - 93 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
            5 - 0.035s 6 (0.000s in global opts, 0.002s io_toposort) - 88 nodes - ('local_fill_sink', 2) ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('local_mul_canonizer', 1)
            6 - 0.038s 10 (0.001s in global opts, 0.002s io_toposort) - 95 nodes - ('local_fill_sink', 3) ('local_dimshuffle_lift', 3) ('constant_folding', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1)
            7 - 0.032s 5 (0.001s in global opts, 0.002s io_toposort) - 91 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1)
            8 - 0.034s 5 (0.000s in global opts, 0.002s io_toposort) - 92 nodes - ('local_fill_sink', 3) ('MergeOptimizer', 1) ('local_greedy_distributor', 1)
            9 - 0.031s 6 (0.001s in global opts, 0.002s io_toposort) - 90 nodes - ('local_fill_sink', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_dimshuffle_lift', 1) ('local_greedy_distributor', 1)
           10 - 0.032s 5 (0.000s in global opts, 0.002s io_toposort) - 89 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('local_fill_sink', 1)
           11 - 0.030s 5 (0.000s in global opts, 0.002s io_toposort) - 88 nodes - ('local_dimshuffle_lift', 2) ('local_fill_to_alloc', 1) ('MergeOptimizer', 1) ('constant_folding', 1)
           12 - 0.026s 1 (0.000s in global opts, 0.003s io_toposort) - 81 nodes - ('MergeOptimizer', 1)
           13 - 0.031s 0 (0.000s in global opts, 0.003s io_toposort) - 81 nodes -
           times - times applied - nb node created - name:
           0.263s - 15 - 0 - constant_folding
           0.096s - 2 - 14 - local_greedy_distributor
           0.066s - 4 - 19 - local_mul_canonizer
           0.046s - 28 - 57 - local_fill_sink
           0.042s - 35 - 78 - local_dimshuffle_lift
           0.018s - 5 - 15 - local_upcast_elemwise_constant_inputs
           0.010s - 11 - 4 - MergeOptimizer
           0.009s - 4 - 0 - local_useless_elemwise
           0.005s - 11 - 2 - local_fill_to_alloc
           0.004s - 3 - 6 - local_neg_to_mul
           0.002s - 1 - 3 - local_lift_transpose_through_dot
           0.002s - 3 - 4 - local_shape_to_shape_i
           0.002s - 2 - 4 - local_subtensor_lift
           0.001s - 3 - 0 - local_subtensor_make_vector
           0.001s - 1 - 1 - local_sum_all_to_none
           0.131s - in 62 optimization that where not used (display only those with a runtime > 0)
             0.050s - local_add_canonizer
             0.018s - local_mul_zero
             0.016s - local_one_minus_erf
             0.010s - local_func_inv
             0.006s - local_0_dot_x
             0.005s - local_track_shape_i
             0.004s - local_mul_switch_sink
             0.004s - local_fill_cut
             0.004s - local_one_minus_erf2
             0.003s - local_remove_switch_const_cond
             0.003s - local_cast_cast
             0.002s - local_IncSubtensor_serialize
             0.001s - local_sum_div_dimshuffle
             0.001s - local_div_switch_sink
             0.001s - local_dimshuffle_no_inplace_at_canonicalize
             0.001s - local_cut_useless_reduce
             0.001s - local_reduce_join
             0.000s - local_sum_sum
             0.000s - local_useless_alloc
             0.000s - local_reshape_chain
             0.000s - local_useless_subtensor
             0.000s - local_reshape_lift
             0.000s - local_flatten_lift
             0.000s - local_useless_slice
             0.000s - local_subtensor_of_alloc
             0.000s - local_subtensor_of_dot
             0.000s - local_subtensor_merge

  * ``0.751816s - ('canonicalize', 'EquilibriumOptimizer', 4) - 0.004s``
    This line is from :class:`SeqOptimizer`, and indicates information related
    to a sub-optimizer. It means that this sub-optimizer took
    a total of .7s. Its name is ``'canonicalize'``. It is an
    :class:`EquilibriumOptimizer`. It was executed at index 4 by the
    :class:`SeqOptimizer`. It spent 0.004s in the *validate* phase.
  * All other lines are from the profiler of the :class:`EquilibriumOptimizer`.

  * An :class:`EquilibriumOptimizer` does multiple passes on the Apply nodes from
    the graph, trying to apply local and global optimizations.
    Conceptually, it tries to execute all global optimizations,
    and to apply all local optimizations on all
    nodes in the graph. If no optimization got applied during a pass, it
    stops. So it tries to find an equilibrium state where none of the
    optimizations get applied. This is useful when we do not know a fixed order for
    the execution of the optimization.
  * ``time 0.751s for 14 passes`` means that it took .7s and did 14 passes over the graph.

  * ``nb nodes (start, end, max) 108 81 117`` means that at the start,
    the graph had 108 node, at the end, it had 81 and the maximum size
    was 117.

  * Then it prints some global timing information: it spent 0.029s in
    :func:`io_toposort`, all local optimizers took 0.687s together for all
    passes, and global optimizers took a total of 0.010s.

  * Then we print the timing for each pass, the optimization that
    got applied, and the number of time they got applied. For example,
    in pass 0, the :func:`local_dimshuffle_lift` optimizer changed the graph 9
    time.

  * Then we print the time spent in each optimizer, the number of times
    they changed the graph and the number of nodes they introduced in
    the graph.

  * Optimizations with that pattern :func:`local_op_lift` means that a node
    with that op will be replaced by another node, with the same op,
    but will do computation closer to the inputs of the graph.
    For instance, ``local_op(f(x))`` getting replaced by ``f(local_op(x))``.

  * Optimization with that pattern :func:`local_op_sink` is the opposite of
    "lift". For instance ``f(local_op(x))`` getting replaced by ``local_op(f(x))``.

  * Local optimizers can replace any arbitrary node in the graph, not
    only the node it received as input. For this, it must return a
    ``dict``. The keys being nodes to replace and the
    values being the corresponding replacement.

    This is useful to replace a client of the node received as
    parameter.

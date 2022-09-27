.. _glossary:

Glossary
========

.. testsetup::

   import aesara
   import aesara.tensor as at

.. glossary::

    Apply
        Instances of :class:`Apply` represent the application of an :term:`Op`
        to some input :term:`Variable` (or variables) to produce some output
        :term:`Variable` (or variables).  They are like the application of a [symbolic]
        mathematical function to some [symbolic] inputs.

    Broadcasting
        Broadcasting is a mechanism which allows tensors with
        different numbers of dimensions to be used in element-by-element
        (i.e. element-wise) computations.  It works by
        (virtually) replicating the smaller tensor along
        the dimensions that it is lacking.

        For more detail, see :ref:`tutbroadcasting`, and also
        * `SciPy documentation about numpy's broadcasting <http://www.scipy.org/EricsBroadcastingDoc>`_
        * `OnLamp article about numpy's broadcasting <http://www.onlamp.com/pub/a/python/2000/09/27/numerically.html>`_

    Constant
        A variable with an immutable value.
        For example, when you type

        >>> x = at.ivector()
        >>> y = x + 3

        Then a `constant` is created to represent the ``3`` in the graph.

        See also: :class:`graph.basic.Constant`


    Elemwise
        An element-wise operation ``f`` on two tensor variables ``M`` and ``N``
        is one such that::

          f(M, N)[i, j] == f(M[i, j], N[i, j])

        In other words, each element of an input matrix is combined
        with the corresponding element of the other(s). There are no
        dependencies between elements whose ``[i, j]`` coordinates do
        not correspond, so an element-wise operation is like a scalar
        operation generalized along several dimensions.  Element-wise
        operations are defined for tensors of different numbers of dimensions by
        :term:`broadcasting` the smaller ones.
        The :class:`Op` responsible for performing element-wise computations
        is :class:`Elemwise`.

    Expression
        See :term:`Apply`

    Expression Graph
        A directed, acyclic set of connected :term:`Variable` and
        :term:`Apply` nodes that express symbolic functional relationship
        between variables.  You use Aesara by defining expression graphs, and
        then compiling them with :term:`aesara.function`.

        See also :term:`Variable`, :term:`Op`, :term:`Apply`, and
        :term:`Type`, or read more about :ref:`graphstructures`.

    Destructive
        An :term:`Op` is destructive--of particular input(s)--if its
        computation requires that one or more inputs be overwritten or
        otherwise invalidated.  For example, :term:`inplace`\ :class:`Op`\s are
        destructive.  Destructive :class:`Op`\s can sometimes be faster than
        non-destructive alternatives.  Aesara encourages users not to put
        destructive :class:`Op`\s into graphs that are given to :term:`aesara.function`,
        but instead to trust the rewrites to insert destructive :class:`Op`\s
        judiciously.

        Destructive :class:`Op`\s are indicated via a :attr:`Op.destroy_map` attribute. (See
        :class:`Op`.


    Graph
        see :term:`expression graph`

    Inplace
        Inplace computations are computations that destroy their inputs as a
        side-effect.  For example, if you iterate over a matrix and double
        every element, this is an inplace operation because when you are done,
        the original input has been overwritten.  :class:`Op`\s representing inplace
        computations are :term:`destructive`, and by default these can only be
        inserted by rewrites, not user code.

    Linker
        A :class:`Linker` instance responsible for "running" the compiled
        function.  Among other things, the linker determines whether
        computations are carried out with
        C or Python code.

    Mode
        A :class:`Mode` instance specifying an :term:`optimizer` and a :term:`linker` that is
        passed to :term:`aesara.function`.  It parametrizes how an expression
        graph is converted to a callable object.

    Op
        The ``.op`` of an :term:`Apply`, together with its symbolic inputs
        fully determines what kind of computation will be carried out for that
        :class:`Apply` at run-time.  Mathematical functions such as addition
        (i.e. :func:`aesara.tensor.add`) and indexing ``x[i]`` are :class:`Op`\s
        in Aesara.  Much of the library documentation is devoted to describing
        the various :class:`Op`\s that are provided with Aesara, but you can add
        more.

        See also :term:`Variable`, :term:`Type`, and :term:`Apply`,
        or read more about :ref:`graphstructures`.

    Rewriter
        A function or class that transforms an Aesara :term:`graph`.

    Optimizer
        An instance of a :term:`rewriter` that has the capacity to provide
        an improvement to the performance of a graph.

    Pure
        An :term:`Op` is *pure* if it has no :term:`destructive` side-effects.

    Storage
        The memory that is used to store the value of a :class:`Variable`.  In most
        cases storage is internal to a compiled function, but in some cases
        (such as :term:`constant` and :term:`shared variable <shared variable>` the storage is not internal.

    Shared Variable
        A :term:`Variable` whose value may be shared between multiple functions.  See :func:`shared <shared.shared>` and :func:`aesara.function <function.function>`.

    aesara.function
        The interface for Aesara's compilation from symbolic expression graphs
        to callable objects.  See :func:`function.function`.

    Type
        The ``.type`` of a
        :term:`Variable` indicates what kinds of values might be computed for it in a
        compiled graph.
        An instance that inherits from :class:`Type`, and is used as the
        ``.type`` attribute of a :term:`Variable`.

        See also :term:`Variable`, :term:`Op`, and :term:`Apply`,
        or read more about :ref:`graphstructures`.

    Variable
        The the main data structure you work with when using Aesara.
        For example,

        >>> x = at.ivector()
        >>> y = -x**2

        ``x`` and ``y`` are both :class:`Variable`\s, i.e. instances of the :class:`Variable` class.

        See also :term:`Type`, :term:`Op`, and :term:`Apply`,
        or read more about :ref:`graphstructures`.

    View
        Some tensor :class:`Op`\s (such as :class:`Subtensor` and :class:`DimShuffle`) can be computed in
        constant time by simply re-indexing their inputs.   The outputs of
        such :class:`Op`\s are views because their
        storage might be aliased to the storage of other variables (the inputs
        of the :class:`Apply`).  It is important for Aesara to know which :class:`Variable`\s are
        views of which other ones in order to introduce :term:`Destructive`
        :class:`Op`\s correctly.

        :class:`Op`\s that are views have their :attr:`Op.view_map` attributes set.

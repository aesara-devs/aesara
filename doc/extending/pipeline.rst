
.. _pipeline:

====================================
Overview of the compilation pipeline
====================================

Once one has an Aesara graph, they can use :func:`aesara.function` to compile a
function that will perform the computations modeled by the graph in Python, C,
Numba, or JAX.

More specifically, :func:`aesara.function` takes a list of input and output
:ref:`Variables <variable>` that define the precise sub-graphs that
correspond to the desired computations.

Here is an overview of the various steps that are taken during the
compilation performed by :func:`aesara.function`.


Step 1 - Create a :class:`FunctionGraph`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The subgraph specified by the end-user is wrapped in a structure called
:class:`FunctionGraph`. This structure defines several callback hooks for when specific
changes are made to a :class:`FunctionGraph`--like adding and
removing nodes, as well as modifying links between nodes
(e.g. modifying an input of an :ref:`apply` node). See :ref:`libdoc_graph_fgraph`.

:class:`FunctionGraph` provides a method to change the input of an :class:`Apply` node from one
:class:`Variable` to another, and a more high-level method to replace a :class:`Variable`
with another. These are the primary means of performing :ref:`graph rewrites <graph_rewriting>`.

Some relevant :ref:`Features <libdoc_graph_fgraphfeature>` are typically added to the
:class:`FunctionGraph` at this stage.  Namely, :class:`Feature`\s that prevent
rewrites from operating in-place on inputs declared as immutable.


Step 2 - Perform graph rewrites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the :class:`FunctionGraph` is constructed, a :term:`rewriter` is produced by
the :term:`mode` passed to :func:`function`. That rewrite is
applied to the :class:`FunctionGraph` using its :meth:`GraphRewriter.rewrite` method.

The rewriter is typically obtained through a query on :attr:`optdb`.


Step 3 - Execute linker to obtain a thunk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the computation graph is rewritten, the :term:`linker` is
extracted from the :class:`Mode`. It is then called with the :class:`FunctionGraph` as
argument to produce a ``thunk``, which is a function with no arguments that
returns nothing. Along with the thunk, one list of input containers (a
:class:`aesara.link.basic.Container` is a sort of object that wraps another and does
type casting) and one list of output containers are produced,
corresponding to the input and output :class:`Variable`\s as well as the updates
defined for the inputs when applicable. To perform the computations,
the inputs must be placed in the input containers, the thunk must be
called, and the outputs must be retrieved from the output containers
where the thunk put them.

Typically, the linker calls the :meth:`FunctionGraph.toposort` method in order to obtain
a linear sequence of operations to perform. How they are linked
together depends on the :class:`Linker` class used. For example, the :class:`CLinker` produces a single
block of C code for the whole computation, whereas the :class:`OpWiseCLinker`
produces one thunk for each individual operation and calls them in
sequence.

The linker is where some options take effect: the ``strict`` flag of
an input makes the associated input container do type checking. The
``borrow`` flag of an output, if ``False``, adds the output to a
``no_recycling`` list, meaning that when the thunk is called the
output containers will be cleared (if they stay there, as would be the
case if ``borrow`` was True, the thunk would be allowed to reuse--or
"recycle"--the storage).

.. note::

    Compiled libraries are stored within a specific compilation directory,
    which by default is set to ``$HOME/.aesara/compiledir_xxx``, where
    ``xxx`` identifies the platform (under Windows the default location
    is instead ``$LOCALAPPDATA\Aesara\compiledir_xxx``). It may be manually set
    to a different location either by setting :attr:`config.compiledir` or
    :attr:`config.base_compiledir`, either within your Python script or by
    using one of the configuration mechanisms described in :mod:`config`.

    The compile cache is based upon the C++ code of the graph to be compiled.
    So, if you change compilation configuration variables, such as
    :attr:`config.blas__ldflags`, you will need to manually remove your compile cache,
    using ``Aesara/bin/aesara-cache clear``

    Aesara also implements a lock mechanism that prevents multiple compilations
    within the same compilation directory (to avoid crashes with parallel
    execution of some scripts).

Step 4 - Wrap the thunk in a pretty package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thunk returned by the linker along with input and output
containers is unwieldy. :func:`aesara.function` hides that complexity away so
that it can be used like a normal function with arguments and return
values.

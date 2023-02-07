
.. _using_modes:

==========================================
Configuration Settings and Compiling Modes
==========================================


Configuration
=============

The :mod:`aesara.config` module contains several *attributes* that modify Aesara's behavior.  Many of these
attributes are examined during the import of the :mod:`aesara` module and several are assumed to be
read-only.

*As a rule, the attributes in the* :mod:`aesara.config` *module should not be modified inside the user code.*

Aesara's code comes with default values for these attributes, but you can
override them from your ``.aesararc`` file, and override those values in turn by
the :envvar:`AESARA_FLAGS` environment variable.

The order of precedence is:

1. an assignment to ``aesara.config.<property>``
2. an assignment in :envvar:`AESARA_FLAGS`
3. an assignment in the ``.aesararc`` file (or the file indicated in :envvar:`AESARARC`)

You can display the current/effective configuration at any time by printing
`aesara.config`.  For example, to see a list  of all active configuration
variables, type this from the command-line:

.. code-block:: bash

    python -c 'import aesara; print(aesara.config)' | less


For more detail, see :ref:`Configuration <libdoc_config>` in the library.


Exercise
========


Consider the logistic regression:

.. testcode::

    import numpy as np
    import aesara
    import aesara.tensor as at


    rng = np.random.default_rng(2498)

    N = 400
    feats = 784
    D = (rng.standard_normal((N, feats)).astype(aesara.config.floatX),
    rng.integers(size=N,low=0, high=2).astype(aesara.config.floatX))
    training_steps = 10000

    # Declare Aesara symbolic variables
    x = at.matrix("x")
    y = at.vector("y")
    w = aesara.shared(rng.standard_normal(feats).astype(aesara.config.floatX), name="w")
    b = aesara.shared(np.asarray(0., dtype=aesara.config.floatX), name="b")
    x.tag.test_value = D[0]
    y.tag.test_value = D[1]

    # Construct Aesara expression graph
    p_1 = 1 / (1 + at.exp(-at.dot(x, w)-b)) # Probability of having a one
    prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
    xent = -y*at.log(p_1) - (1-y)*at.log(1-p_1) # Cross-entropy
    cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
    gw,gb = at.grad(cost, [w,b])

    # Compile expressions to functions
    train = aesara.function(
        inputs=[x,y],
        outputs=[prediction, xent],
        updates=[(w, w-0.01*gw), (b, b-0.01*gb)],
        name = "train"
    )
    predict = aesara.function(
        inputs=[x], outputs=prediction,
        name = "predict"
    )

    if any(x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm']
           for x in train.maker.fgraph.toposort()):
        print('Used the cpu')
    else:
        print('ERROR, not able to tell if aesara used the cpu or another device')
        print(train.maker.fgraph.toposort())

    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print("target values for D")
    print(D[1])

    print("prediction on D")
    print(predict(D[0]))

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   Used the cpu
   target values for D
   ...
   prediction on D
   ...

Modify and execute this example to run on CPU (the default) with ``floatX=float32`` and
time the execution using the command line ``time python file.py``.  Save your code
as it will be useful later on.

.. Note::

   * Apply the Aesara flag ``floatX=float32`` (through ``aesara.config.floatX``) in your code.
   * Cast inputs before storing them into a shared variable.
   * Circumvent the automatic cast of int32 with float32 to float64:

     * Insert manual cast in your code or use [u]int{8,16}.
     * Insert manual cast around the mean operator (this involves division by length, which is an int64).
     * Note that a new casting mechanism is being developed.

:download:`Solution<modes_solution_1.py>`

-------------------------------------------

Default Modes
=============

Every time :func:`aesara.function <function.function>` is called,
the symbolic relationships between the input and output Aesara *variables*
are rewritten and compiled. The way this compilation occurs
is controlled by the value of the ``mode`` parameter.

Aesara defines the following modes by name:

- ``'FAST_COMPILE'``: Apply just a few graph optimizations and only use Python implementations.
- ``'FAST_RUN'``: Apply all optimizations and use C implementations where possible.
- ``'DebugMode'``: Verify the correctness of all optimizations, and compare C and Python
    implementations. This mode can take much longer than the other modes, but can identify
    several kinds of problems.
- ``'NanGuardMode'``: Same optimization as FAST_RUN, but :ref:`check if a node generate nans. <nanguardmode>`

The default mode is typically ``FAST_RUN``, but it can be controlled via
the configuration variable :attr:`config.mode`,
which can be overridden by passing the keyword argument to
:func:`aesara.function <function.function>`.

================= =============================================================== ===============================================================================
short name        Full constructor                                                What does it do?
================= =============================================================== ===============================================================================
``FAST_COMPILE``  ``compile.mode.Mode(linker='py', optimizer='fast_compile')``    Python implementations only, quick and cheap graph transformations
``FAST_RUN``      ``compile.mode.Mode(linker='cvm', optimizer='fast_run')``       C implementations where available, all available graph transformations.
``DebugMode``     ``compile.debugmode.DebugMode()``                               Both implementations where available, all available graph transformations.
================= =============================================================== ===============================================================================

.. Note::

    For debugging purpose, there also exists a :class:`MonitorMode` (which has no
    short name). It can be used to step through the execution of a function:
    see :ref:`the debugging FAQ<faq_monitormode>` for details.


Default Linkers
===============

A :class:`Mode` object is composed of two things: an optimizer and a linker. Some modes,
like `NanGuardMode` and `DebugMode`, add logic around the
optimizer and linker. `DebugMode` uses its own linker.

You can select which linker to use with the Aesara flag :attr:`config.linker`.
Here is a table to compare the different linkers.

=============  =========  =================  =========  ===
linker         gc [#gc]_  Raise error by op  Overhead   Definition
=============  =========  =================  =========  ===
cvm            yes        yes                "++"       As c|py, but the runtime algo to execute the code is in c
cvm_nogc       no         yes                "+"        As cvm, but without gc
c|py [#cpy1]_  yes        yes                "+++"      Try C code. If none exists for an op, use Python
c|py_nogc      no         yes                "++"       As c|py, but without gc
c              no         yes                "+"        Use only C code (if none available for an op, raise an error)
py             yes        yes                "+++"      Use only Python code
NanGuardMode   yes        yes                "++++"     Check if nodes generate NaN
DebugMode      no         yes                VERY HIGH  Make many checks on what Aesara computes
=============  =========  =================  =========  ===


.. [#gc] Garbage collection of intermediate results during computation.
         Otherwise, their memory space used by the ops is kept between
         Aesara function calls, in order not to
         reallocate memory, and lower the overhead (make it faster...).
.. [#cpy1] Default


For more detail, see :ref:`Mode<libdoc_compile_mode>` in the library.

.. _optimizers:

Default Optimizers
==================

Aesara allows compilations with a number of predefined rewrites that are
expected to improve graph evaluation performance on average.
An optimizer is technically just a :class:`Rewriter`, or an object that
indicates a particular set of rewrites (e.g. a string used to query `optdb` for
a :class:`Rewriter`).

The optimizers Aesara provides are summarized below to indicate the trade-offs
one might make between compilation time and execution time.

These optimizers can be enabled globally with the Aesara flag: ``optimizer=name``
or per call to aesara functions with ``function(...mode=Mode(optimizer="name"))``.

=================  ============  ==============  ==================================================
optimizer          Compile time  Execution time  Description
=================  ============  ==============  ==================================================
None               "++++++"      "+"             Applies none of Aesara's rewrites
o1 (fast_compile)  "+++++"       "++"            Applies only basic rewrites
o2                 "++++"        "+++"           Applies few basic rewrites and some that compile fast
o3                 "+++"         "++++"          Applies all rewrites except ones that compile slower
o4 (fast_run)      "++"          "+++++"         Applies all rewrites
unsafe             "+"           "++++++"        Applies all rewrites, and removes safety checks
stabilize          "+++++"       "++"            Only applies stability rewrites
=================  ============  ==============  ==================================================

For a detailed list of the specific rewrites applied for each of these
optimizers, see :ref:`optimizations`. Also, see :ref:`unsafe_rewrites` and
:ref:`faster-aesara-function-compilation` for other trade-off.


.. _using_debugmode:

Using :class:`DebugMode`
========================

While normally you should use the ``FAST_RUN`` or ``FAST_COMPILE`` mode,
it is useful at first--especially when you are defining new kinds of
expressions or new rewrites--to run your code using the `DebugMode`
(available via ``mode='DebugMode``). The `DebugMode` is designed to
run several self-checks and assertions that can help diagnose
possible programming errors leading to incorrect output. Note that
`DebugMode` is much slower than ``FAST_RUN`` or ``FAST_COMPILE``, so
use it only during development.

.. If you modify this code, also change :
.. tests/test_tutorial.py:T_modes.test_modes_1

`DebugMode` is used as follows:

.. testcode::

    x = at.dvector('x')

    f = aesara.function([x], 10 * x, mode='DebugMode')

    f([5])
    f([0])
    f([7])


If any problem is detected, `DebugMode` will raise an exception according to
what went wrong, either at call time (e.g. ``f(5)``) or compile time (
``f = aesara.function(x, 10 * x, mode='DebugMode')``). These exceptions
should *not* be ignored; talk to your local Aesara guru or email the
users list if you cannot make the exception go away.

Some kinds of errors can only be detected for certain input value combinations.
In the example above, there is no way to guarantee that a future call to, say
``f(-1)``, won't cause a problem.  `DebugMode` is not a silver bullet.

.. TODO: repair the following link

If you instantiate `DebugMode` using the constructor (see :class:`DebugMode`)
rather than the keyword `DebugMode` you can configure its behaviour via
constructor arguments. The keyword version of `DebugMode` (which you get by using ``mode='DebugMode'``)
is quite strict.

For more detail, see :ref:`DebugMode<debugmode>` in the library.

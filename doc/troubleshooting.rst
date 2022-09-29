.. _troubleshooting:

Troubleshooting
###############

Here are Linux troubleshooting instructions. There is a specific `MacOS`_ section.

- :ref:`network_error_proxy`
- :ref:`slow_or_memory`
- :ref:`TensorVariable_TypeError`
- :ref:`out_of_memory`
- :ref:`float64_output`
- :ref:`test_aesara`
- :ref:`test_BLAS`

.. _network_error_proxy:

Why do I get a network error when I install Aesara
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are behind a proxy, you must do some extra configuration steps
before starting the installation. You must set the environment
variable ``http_proxy`` to the proxy address. Using bash this is
accomplished with the command
``export http_proxy="http://user:pass@my.site:port/"``
You can also provide the ``--proxy=[user:pass@]url:port`` parameter
to pip. The ``[user:pass@]`` portion is optional.

.. _TensorVariable_TypeError:

How to solve TypeError: object of type 'TensorVariable' has no len()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you receive the following error, it is because the Python function *__len__* cannot
be implemented on Aesara variables:

.. code-block:: python

   TypeError: object of type 'TensorVariable' has no len()

Python requires that *__len__* returns an integer, yet it cannot be done as Aesara's variables are symbolic. However, `var.shape[0]` can be used as a workaround.

This error message cannot be made more explicit because the relevant aspects of Python's
internals cannot be modified.

.. _out_of_memory:

How to solve Out of memory Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Occasionally Aesara may fail to allocate memory when there appears to be more
than enough reporting:

    Error allocating X bytes of device memory (out of memory). Driver report Y
    bytes free and Z total.

where X is far less than Y and Z (i.e. X << Y < Z).

This scenario arises when an operation requires allocation of a large contiguous
block of memory but no blocks of sufficient size are available.

A known example is related to writing data to shared variables. When updating a
shared variable Aesara will allocate new space if the size of the data does not
match the size of the space already assigned to the variable. This can lead to
memory fragmentation which means that a continugous block of memory of
sufficient capacity may not be available even if the free memory overall is
large enough.

.. _float64_output:

aesara.function returns a float64 when the inputs are float32 and int{32, 64}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It should be noted that using float32 and int{32, 64} together
inside a function would provide float64 as output.

To help you find where float64 are created, see the
:attr:`warn_float64` Aesara flag.

.. _test_aesara:

How to test that Aesara works properly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An easy way to check something that could be wrong is by making sure ``AESARA_FLAGS``
have the desired values as well as the ``~/.aesararc``

Also, check the following outputs :

.. code-block:: bash

    ipython

.. code-block:: python

    import aesara
    aesara.__file__
    aesara.__version__


Once you have installed Aesara, you should run the test suite in the ``tests`` directory.

.. code-block:: bash

    python -c "import numpy; numpy.test()"
    python -c "import scipy; scipy.test()"
    pip install pytest
    AESARA_FLAGS='' pytest tests/

All Aesara tests should pass (skipped tests and known failures are normal). If
some test fails on your machine, you are encouraged to tell us what went
wrong in the GitHub issues.

.. _slow_or_memory:

Why is my code so slow/uses so much memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a few things you can easily do to change the trade-off
between speed and memory usage.

Could raise memory usage but speed up computation:

- :attr:`config.allow_gc` =False

Could lower the memory usage, but raise computation time:

- :attr:`config.scan__allow_gc` = True
- :attr:`config.scan__allow_output_prealloc` =False
- Disable one or scan more rewrites:
    - ``optimizer_excluding=scan_pushout_seqs_ops``
    - ``optimizer_excluding=scan_pushout_dot1``
    - ``optimizer_excluding=scan_pushout_add``
- Disable all rewrites tagged as raising memory usage:
  ``optimizer_excluding=more_mem`` (currently only the 3 scan rewrites above)
- `float16 <https://github.com/Theano/Theano/issues/2908>`_.

If you want to analyze the memory usage during computation, the
simplest is to let the memory error happen during Aesara execution and
use the Aesara flags :attr:`exception_verbosity=high`.

.. _test_BLAS:

How do I configure/test my BLAS library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are many ways to configure BLAS for Aesara. This is done with the Aesara
flags ``blas__ldflags`` (:ref:`libdoc_config`). The default is to use the BLAS
installation information in NumPy, accessible via
``numpy.__config__.show()``.  You can tell aesara to use a different
version of BLAS, in case you did not compile NumPy with a fast BLAS or if NumPy
was compiled with a static library of BLAS (the latter is not supported in
Aesara).

The short way to configure the Aesara flags ``blas__ldflags`` is by setting the
environment variable :envvar:`AESARA_FLAGS` to ``blas__ldflags=XXX`` (in bash
``export AESARA_FLAGS=blas__ldflags=XXX``)

The ``${HOME}/.aesararc`` file is the simplest way to set a relatively
permanent option like this one.  Add a ``[blas]`` section with an ``ldflags``
entry like this:

.. code-block:: cfg

    # other stuff can go here
    [blas]
    ldflags = -lf77blas -latlas -lgfortran #put your flags here

    # other stuff can go here

For more information on the formatting of ``~/.aesararc`` and the
configuration options that you can put there, see :ref:`libdoc_config`.

Here are some different way to configure BLAS:

0) Do nothing and use the default config, which is to link against the same
BLAS against which NumPy was built. This does not work in the case NumPy was
compiled with a static library (e.g. ATLAS is compiled by default only as a
static library).

1) Disable the usage of BLAS and fall back on NumPy for dot products. To do
this, set the value of ``blas__ldflags`` as the empty string (ex: ``export
AESARA_FLAGS=blas__ldflags=``). Depending on the kind of matrix operations your
Aesara code performs, this might slow some things down (vs. linking with BLAS
directly).

2) You can install the default (reference) version of BLAS if the NumPy version
(against which Aesara links) does not work. If you have root or sudo access in
fedora you can do ``sudo yum install blas blas-devel``. Under Ubuntu/Debian
``sudo apt-get install libblas-dev``. Then use the Aesara flags
``blas__ldflags=-lblas``. Note that the default version of blas is not optimized.
Using an optimized version can give up to 10x speedups in the BLAS functions
that we use.

3) Install the ATLAS library. ATLAS is an open source optimized version of
BLAS. You can install a precompiled version on most OSes, but if you're willing
to invest the time, you can compile it to have a faster version (we have seen
speed-ups of up to 3x, especially on more recent computers, against the
precompiled one). On Fedora, ``sudo yum install atlas-devel``. Under Ubuntu,
``sudo apt-get install libatlas-base-dev libatlas-base`` or
``libatlas3gf-sse2`` if your CPU supports SSE2 instructions. Then set the
Aesara flags ``blas__ldflags`` to ``-lf77blas -latlas -lgfortran``. Note that
these flags are sometimes OS-dependent.

4) Use a faster version like MKL, GOTO, ... You are on your own to install it.
See the doc of that software and set the Aesara flags ``blas__ldflags``
correctly (for example, for MKL this might be ``-lmkl -lguide -lpthread`` or
``-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lguide -liomp5 -lmkl_mc
-lpthread``).

.. note::

    Make sure your BLAS
    libraries are available as dynamically-loadable libraries.
    ATLAS is often installed only as a static library.  Aesara is not able to
    use this static library. Your ATLAS installation might need to be modified
    to provide dynamically loadable libraries.  (On Linux this
    typically means a library whose name ends with .so. On Windows this will be
    a .dll, and on OS-X it might be either a .dylib or a .so.)

    This might be just a problem with the way Aesara passes compilation
    arguments to g++, but the problem is not fixed yet.

.. note::

    If you have problems linking with MKL, `Intel Line Advisor
    <http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor>`_
    and the `MKL User Guide
    <http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_userguide_lnx/index.htm>`_
    can help you find the correct flags to use.

.. note::

    If you have error that contain "gfortran" in it, like this one:

        ImportError: ('/home/Nick/.aesara/compiledir_Linux-2.6.35-31-generic-x86_64-with-Ubuntu-10.10-maverick--2.6.6/tmpIhWJaI/0c99c52c82f7ddc775109a06ca04b360.so: undefined symbol: _gfortran_st_write_done'

    The problem is probably that NumPy is linked with a different blas
    then then one currently available (probably ATLAS). There is 2
    possible fixes:

    1) Uninstall ATLAS and install OpenBLAS.
    2) Use the Aesara flag "blas__ldflags=-lblas -lgfortran"

    1) is better as OpenBLAS is faster then ATLAS and NumPy is
    probably already linked with it. So you won't need any other
    change in Aesara files or Aesara configuration.

Testing BLAS
------------

It is recommended to test your Aesara/BLAS integration. There are many versions
of BLAS that exist and there can be up to 10x speed difference between them.
Also, having Aesara link directly against BLAS instead of using NumPy/SciPy as
an intermediate layer reduces the computational overhead. This is
important for BLAS calls to ``ger``, ``gemv`` and small ``gemm`` operations
(automatically called when needed when you use ``dot()``). To run the
Aesara/BLAS speed test:

.. code-block:: bash

    python `python -c "import os, aesara; print(os.path.dirname(aesara.__file__))"`/misc/check_blas.py

This will print a table with different versions of BLAS/numbers of
threads on multiple CPUs. It will also print some Aesara/NumPy
configuration information. Then, it will print the running time of the same
benchmarks for your installation. Try to find a CPU similar to yours in
the table, and check that the single-threaded timings are roughly the same.

Aesara should link to a parallel version of Blas and use all cores
when possible. By default it should use all cores. Set the environment
variable "OMP_NUM_THREADS=N" to specify to use N threads.


.. _MacOS:

Mac OS
------

Although the above steps should be enough, running Aesara on a Mac may
sometimes cause unexpected crashes, typically due to multiple versions of
Python or other system libraries. If you encounter such problems, you may
try the following.

- You can ensure MacPorts shared libraries are given priority at run-time
  with ``export LD_LIBRARY_PATH=/opt/local/lib:$LD_LIBRARY_PATH``. In order
  to do the same at compile time, you can add to your ``~/.aesararc``:

    .. code-block:: cfg

      [gcc]
      cxxflags = -L/opt/local/lib

- More generally, to investigate libraries issues, you can use the ``otool -L``
  command on ``.so`` files found under your ``~/.aesara`` directory. This will
  list shared libraries dependencies, and may help identify incompatibilities.

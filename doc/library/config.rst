.. _libdoc_config:

=======================================
:mod:`config` -- Aesara Configuration
=======================================

.. module:: config
   :platform: Unix, Windows
   :synopsis: Library configuration attributes.
.. moduleauthor:: LISA


Guide
=====

The config module contains many ``attributes`` that modify Aesara's behavior.  Many of these
attributes are consulted during the import of the ``aesara`` module, and some
are assumed to be read-only.

*As a rule, the attributes in this module should not be modified by user code.*

Aesara's code comes with default values for these attributes, but they can be
overridden via a user's ``.aesararc`` file and the :envvar:`AESARA_FLAGS`
environment variable.

The order of precedence is:

1. an assignment to ``aesara.config.<property>``
2. :envvar:`AESARA_FLAGS`
3. the ``.aesararc`` file (or the file indicated by :envvar:`AESARARC`)

The current/effective configuration can be shown at any time by printing the
object ``aesara.config``.  For example, to see a list of all active
configuration variables, type this from the command-line:

.. code-block:: bash

    python -c 'import aesara; print(aesara.config)' | less

Environment Variables
=====================


.. envvar:: AESARA_FLAGS

    This is a list of comma-delimited ``key=value`` pairs that control
    Aesara's behavior.

    For example, in ``bash``, one can override the :envvar:`AESARARC` defaults
    for a script ``<myscript>.py`` with the following:

    .. code-block:: bash

        AESARA_FLAGS='floatX=float32'  python <myscript>.py

    If a value is defined several times in ``AESARA_FLAGS``,
    the right-most definition is used, so, for instance, if
    ``AESARA_FLAGS='floatX=float32,floatX=float64'`` is set, then ``float64`` will be
    used.

.. envvar:: AESARARC

    The location(s) of the ``.aesararc`` file(s) in `ConfigParser` format.
    It defaults to ``$HOME/.aesararc``. On Windows, it defaults to
    ``$HOME/.aesararc:$HOME/.aesararc.txt``.

    Here is the ``.aesararc`` equivalent to the ``AESARA_FLAGS`` in the example above:

    .. code-block:: cfg

        [global]
        floatX = float32
        device = cuda0

    Configuration attributes that are available directly in ``config``
    (e.g. ``config.mode``) should be defined in the ``[global]`` section.
    Attributes from a subsection of ``config``
    (e.g. ``config.dnn__conv__algo_fwd``) should be defined in their
    corresponding section (e.g. ``[dnn.conv]``).

    Multiple configuration files can be specified by separating them with ``':'``
    characters (as in ``$PATH``).  Multiple configuration files will be merged,
    with later (right-most) files taking priority over earlier files, when
    multiple files specify values for the same configuration option.

    For example, to override system-wide settings with personal ones,
    set ``AESARARC=/etc/aesararc:~/.aesararc``. To load configuration files in
    the current working directory, append ``.aesararc`` to the list of configuration
    files, e.g. ``AESARARC=~/.aesararc:.aesararc``.

Config Attributes
=====================

The list below describes some of the more common and important flags
that you might want to use. For the complete list (including documentation),
import ``aesara`` and print the config variable, as in:

.. code-block:: bash

    python -c 'import aesara; print(aesara.config)' | less

.. attribute:: device

    String value: either ``'cpu'``

.. attribute:: force_device

    Bool value: either ``True`` or ``False``

    Default: ``False``

    This flag's value cannot be modified during the program execution.

.. attribute:: print_active_device

    Bool value: either ``True`` or ``False``

    Default: ``True``

    Print the active device when the device is initialized.

.. attribute:: floatX

    String value: ``'float64'``, ``'float32'``, or ``'float16'`` (with limited support)

    Default: ``'float64'``

    This sets the default dtype returned by ``tensor.matrix()``, ``tensor.vector()``,
    and similar functions.  It also sets the default Aesara bit width for
    arguments passed as Python floating-point numbers.

.. attribute:: warn_float64

    String value: either ``'ignore'``, ``'warn'``, ``'raise'``, or ``'pdb'``

    Default: ``'ignore'``

    This option determines what's done when a :class:`TensorVariable` with dtype
    equal to ``float64`` is created.
    This can be used to help find upcasts to ``float64`` in user code.

.. attribute:: deterministic

    String value: either ``'default'``, ``'more'``

    Default: ``'default'``

    If ``more``, sometimes Aesara will select :class:`Op` implementations that
    are more "deterministic", but slower.  See the ``dnn.conv.algo*``
    flags for more cases.

.. attribute:: allow_gc

    Bool value: either ``True`` or ``False``

    Default: ``True``

    This determines whether or not Aesara's garbage collector is used for
    intermediate results. To use less memory, Aesara frees the intermediate
    results as soon as they are no longer needed.  Disabling Aesara's garbage
    collection allows Aesara to reuse buffers for intermediate results between
    function calls. This speeds up Aesara by spending less time reallocating
    space during function evaluation and can provide significant speed-ups for
    functions with many fast :class:`Op`\s, but it also increases Aesara's memory
    usage.

.. attribute:: config.scan__allow_output_prealloc

    Bool value, either ``True`` or ``False``

    Default: ``True``

    This enables, or disables, a rewrite in :class:`Scan` that tries to
    pre-allocate memory for its outputs. Enabling the rewrite can give a
    significant speed up at the cost of slightly increased memory usage.

.. attribute:: config.scan__allow_gc

    Bool value, either ``True`` or ``False``

    Default: ``False``

    Allow garbage collection inside of :class:`Scan` :class:`Op`\s.

    If :attr:`config.allow_gc` is ``True``, but :attr:`config.scan__allow_gc` is
    ``False``, then Aesara will perform garbage collection during the inner
    operations of a :class:`Scan` after each iterations.

.. attribute:: cycle_detection

    String value, either ``regular`` or ``fast```

    Default: ``regular``

    If :attr:`cycle_detection` is set to ``regular``, most in-place operations are allowed,
    but graph compilation is slower. If :attr:`cycle_detection` is set to ``faster``,
    less in-place operations are allowed, but graph compilation is faster.

.. attribute:: check_stack_trace

    String value, either ``off``, ``log``, ``warn``, ``raise``

    Default: ``off``

    This is a flag for checking stack traces during graph rewriting.
    If :attr:`check_stack_trace` is set to ``off``, no check is performed on the
    stack trace. If :attr:`check_stack_trace` is set to ``log`` or ``warn``, a
    dummy stack trace is inserted that indicates which rewrite inserted the
    variable that had an empty stack trace, but, when ``warn`` is set, a warning
    is also printed.
    If :attr:`check_stack_trace` is set to ``raise``, an exception is raised if a
    stack trace is missing.

.. attribute:: openmp

    Bool value: either ``True`` or ``False``

    Default: ``False``

    Enable or disable parallel computation on the CPU with OpenMP.
    It is the default value used by :class:`Op`\s that support OpenMP.
    It is best to specify this setting in ``.aesararc`` or in the environment
    variable ``AESARA_FLAGS``.

.. attribute:: openmp_elemwise_minsize

    Positive int value, default: 200000.

    This specifies the minimum size of a vector for which OpenMP will be used by
    :class:`Elemwise` :class:`Op`\s, when OpenMP is enabled.

.. attribute:: cast_policy

    String value: either ``'numpy+floatX'`` or ``'custom'``

    Default: ``'custom'``

    This specifies how data types are implicitly determined by Aesara during the
    creation of constants or in the results of arithmetic operations.

    The ``'custom'`` value corresponds to a set of custom rules originally used
    in Aesara.  These rules can be partially customized (e.g. see the in-code
    help of ``aesara.scalar.basic.NumpyAutocaster``).  This will be deprecated
    in the future.

    The ``'numpy+floatX'`` setting attempts to mimic NumPy casting rules,
    although it prefers to use ``float32` `numbers instead of ``float64`` when
    ``config.floatX`` is set to ``'float32'`` and the associated data is not
    explicitly typed as ``float64`` (e.g. regular Python floats).  Note that
    ``'numpy+floatX'`` is not currently behaving exactly as planned (it is a
    work-in-progress), and thus it should considered experimental.

    At the moment it behaves differently from NumPy in the following situations:

    * Depending on the value of :attr:`config.int_division`, the resulting dtype
      of a division of integers with the ``/`` operator may not match
      that of NumPy.
    * On mixed scalar and array operations, NumPy tries to prevent the scalar
      from upcasting the array's type unless it is of a fundamentally
      different type. Aesara does not attempt to do the same at this point,
      so users should be careful, since scalars may upcast arrays when they
      otherwise wouldn't in NumPy. This behavior should change in the near
      future.

.. attribute:: int_division

    String value: either ``'int'``, ``'floatX'``, or ``'raise'``

    Default: ``'int'``

    Specifies what to do when one tries to compute ``x / y``, where both ``x`` and
    ``y`` are of integer types (possibly unsigned). ``'int'`` means an integer is
    returned (as in Python 2.X). This behavior is deprecated.

    ``'floatX'`` returns a number of with the dtype given by ``config.floatX``.

    ``'raise'`` is the safest choice (and will become default in a future
    release of Aesara).  It raises an error when one tries to perform such an
    operation, enforcing the use of the integer division operator (``//``). If a
    float result is desired, either cast one of the arguments to a float, or use
    ``x.__truediv__(y)``.

.. attribute:: mode

    String value: ``'Mode'``, ``'DebugMode'``, ``'FAST_RUN'``,
    ``'FAST_COMPILE'``

    Default: ``'Mode'``

    This sets the default compilation mode when compiling Aesara functions. By
    default the mode ``'Mode'`` is equivalent to ``'FAST_RUN'``.

.. attribute:: profile

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, the VM and CVM linkers profile the execution time of Aesara functions.

    See :ref:`tut_profiling` for examples.

.. attribute:: profile_memory

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, the VM and CVM linkers profile the memory usage of Aesara
    functions.  This only works when ``profile=True``.

.. attribute:: profile_optimizer

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, the :class:`VM` and :class:`CVM` linkers profile the rewriting phase when
    compiling an Aesara function.  This only works when ``profile=True``.

.. attribute:: config.profiling__n_apply

    Positive int value, default: 20.

    The number of :class:`Apply` nodes to print in the profiler output.

.. attribute:: config.profiling__n_ops

    Positive int value, default: 20.

    The number of :class:`Op`\s to print in the profiler output.

.. attribute:: config.profiling__min_memory_size

    Positive int value, default: 1024.

    During memory profiling, do not print :class:`Apply` nodes if the size
    of their outputs (in bytes) is lower than this value.

.. attribute:: config.profiling__min_peak_memory

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, print the minimum peak memory usage during memory profiling.
    This only works when ``profile=True`` and ``profile_memory=True``.

.. attribute:: config.profiling__destination

    String value: ``'stderr'``, ``'stdout'``, or a name of a
    file to be created

    Default: ``'stderr'``

    Name of the destination file for the profiling output.
    The profiling output can be directed to stderr (default), stdout, or an
    arbitrary file.

.. attribute:: config.profiling__debugprint

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, use ``debugprint`` to print the profiled functions.

.. attribute:: config.profiling__ignore_first_call

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, ignore the first call to an Aesara function while profiling.

.. attribute:: config.lib__amblibm

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, use the `amdlibm
    <https://developer.amd.com/amd-cpu-libraries/amd-math-library-libm/>`__
    library, which is faster than the standard ``libm``.

.. attribute:: linker

    String value: ``'c|py'``, ``'py'``, ``'c'``, ``'c|py_nogc'``

    Default: ``'c|py'``

    When the mode is ``'Mode'``, it sets the default linker used.
    See :ref:`using_modes` for a comparison of the different linkers.

.. attribute:: optimizer

    String value: ``'fast_run'``, ``'merge'``, ``'fast_compile'``, ``'None'``

    Default: ``'fast_run'``

    When the mode is ``'Mode'``, it sets the default rewrites used during compilation.

.. attribute:: on_opt_error

    String value: ``'warn'``, ``'raise'``, ``'pdb'`` or ``'ignore'``

    Default: ``'warn'``

    When a crash occurs while trying to apply a rewrite, either warn the
    user and skip the rewrite (i.e. ``'warn'``), raise the exception
    (i.e. ``'raise'``), drop into the ``pdb`` debugger (i.e. ``'pdb'``), or
    ignore it (i.e. ``'ignore'``).
    We suggest never using ``'ignore'`` except during testing.

.. attribute:: assert_no_cpu_op

    String value: ``'ignore'`` or ``'warn'`` or ``'raise'`` or ``'pdb'``

    Default: ``'ignore'``

    If there is a CPU :class:`Op` in the computational graph, depending on its value,
    this flag can either raise a warning, an exception or drop into the frame
    with ``pdb``.

.. attribute:: on_shape_error

    String value: ``'warn'`` or ``'raise'``

    Default: ``'warn'``

    When an exception is raised while inferring the shape of an :class:`Apply`
    node, either warn the user and use a default value (i.e. ``'warn'``), or
    raise the exception (i.e. ``'raise'``).


.. attribute:: config.warn__ignore_bug_before

    String value: ``'None'``, ``'all'``, ``'0.3'``, ``'0.4'``, ``'0.4.1'``,
    ``'0.5'``, ``'0.6'``, ``'0.7'``, ``'0.8'``, ``'0.8.1'``, ``'0.8.2'``,
    ``'0.9'``, ``'0.10'``, ``'1.0'``, ``'1.0.1'``, ``'1.0.2'``, ``'1.0.3'``,
    ``'1.0.4'``,``'1.0.5'``

    Default: ``'0.9'``

    When we an Aesara bug that generated a bad result is fixed, we also make
    Aesara raise a warning when it encounters the same circumstances again. This
    helps users determine whether or not said bug has affected past runs, since
    one only needs to perform the same runs again with the new version, and one
    does not have to understand the Aesara internals that triggered the bug.

    This flag lets users ignore warnings about old bugs that were
    fixed before their first checkout of Aesara.
    You can set its value to the first version of Aesara
    that you used (probably 0.3 or higher)

    ``'None'`` means that all warnings will be displayed.
    ``'all'`` means all warnings will be ignored.

    This flag's value cannot be modified during program execution.

.. attribute:: base_compiledir

    Default: On Windows: ``$LOCALAPPDATA\\Aesara`` if ``$LOCALAPPDATA`` is defined,
    otherwise and on other systems: ``~/.aesara``.

    This directory stores the platform-dependent compilation directories.

    This flag's value cannot be modified during program execution.

.. attribute:: compiledir_format

    Default: ``"compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s"``

    This is a Python format string that specifies the sub-directory of
    ``config.base_compiledir`` in which platform-dependent compiled modules are
    stored. To see a list of all available substitution keys, run ``python -c
    "import aesara; print(aesara.config)"`` and look for ``compiledir_format``.

    This flag's value cannot be modified during program execution.

.. attribute:: compiledir

    Default: ``config.base_compiledir``/``config.compiledir_format``

    This directory stores dynamically-compiled modules for a particular
    platform.

    This flag's value cannot be modified during program execution.

.. attribute:: config.blas__ldflags

    Default: ``'-lblas'``

    Link argument to link against a (Fortran) level-3 blas implementation.
    Aesara will test if ``'-lblas'`` works by default. If not, it will disable C
    code for BLAS.

.. attribute:: config.experimental__local_alloc_elemwise_assert

    Bool value: either ``True`` or ``False``

    Default: ``True``

    When ``True``, add asserts that highlight shape errors.

    Without such asserts, the underlying rewrite could hide errors in user
    code.  Aesara adds the asserts only if it cannot infer that the shapes are
    equivalent.  When it can determine equivalence, this rewrite does not
    introduce an assert.

    Removing these asserts can speed up execution.

.. attribute:: config.dnn__enabled

    String value: ``'auto'``, ``'True'``, ``'False'``, ``'no_check'``

    Default: ``'auto'``

    If ``'auto'``, automatically detect and use
    `cuDNN <https://developer.nvidia.com/cudnn>`_ when it is available.
    If cuDNN is unavailable, do not raise an error.

    If ``'True'``, require the use of cuDNN.  If cuDNN is unavailable, raise an error.

    If ``'False'``, neither use cuDNN nor check if it is available.

    If ``'no_check'``, assume cuDNN is present and that the versions between the
    header and library match.

.. attribute:: config.dnn__include_path

    Default: ``include`` sub-folder in CUDA root directory, or headers paths defined for the compiler.

    Location of the cuDNN header.

.. attribute:: config.dnn__library_path

    Default: Library sub-folder (``lib64`` on Linux) in CUDA root directory, or
    libraries paths defined for the compiler.

    Location of the cuDNN library.

.. attribute:: config.conv__assert_shape

    If ``True``, ``AbstractConv*`` :class:`Op`\s will verify that user-provided shapes
    match the run-time shapes. This is a debugging option, and may slow down
    compilation.

.. attribute:: config.dnn.conv.workmem

    Deprecated, use :attr:`config.dnn__conv__algo_fwd`.


.. attribute:: config.dnn.conv.workmem_bwd

    Deprecated, use :attr:`config.dnn__conv__algo_bwd_filter` and
    :attr:`config.dnn__conv__algo_bwd_data` instead.

.. attribute:: config.dnn__conv__algo_fwd

    String value:
    ``'small'``, ``'none'``, ``'large'``, ``'fft'``, ``'fft_tiling'``,
    ``'winograd'``, ``'winograd_non_fused'``, ``'guess_once'``, ``'guess_on_shape_change'``,
    ``'time_once'``, ``'time_on_shape_change'``.

    Default: ``'small'``

    3d convolution only support ``'none'``, ``'small'``, ``'fft_tiling'``, ``'guess_once'``,
    ``'guess_on_shape_change'``, ``'time_once'``, ``'time_on_shape_change'``.

.. attribute:: config.dnn.conv.algo_bwd

    Deprecated, use :attr:`config.dnn__conv__algo_bwd_filter` and
    :attr:`config.dnn__conv__algo_bwd_data` instead.

.. attribute:: config.dnn__conv__algo_bwd_filter

    String value:
    ``'none'``, ``'deterministic'``, ``'fft'``, ``'small'``, ``'winograd_non_fused'``, ``'fft_tiling'``, ``'guess_once'``,
    ``'guess_on_shape_change'``, ``'time_once'``, ``'time_on_shape_change'``.

    Default: ``'none'``

    3d convolution only supports ``'none'``, ``'small'``, ``'guess_once'``,
    ``'guess_on_shape_change'``, ``'time_once'``, ``'time_on_shape_change'``.

.. attribute:: config.dnn__conv__algo_bwd_data

    String value:
    ``'none'``, ``'deterministic'``, ``'fft'``, ``'fft_tiling'``, ``'winograd'``,
    ``'winograd_non_fused'``, ``'guess_once'``, ``'guess_on_shape_change'``, ``'time_once'``,
    ``'time_on_shape_change'``.

    Default: ``'none'``

    3d convolution only supports ``'none'``, ``'deterministic'``, ``'fft_tiling'``
    ``'guess_once'``, ``'guess_on_shape_change'``, ``'time_once'``,
    ``'time_on_shape_change'``.

.. attribute:: config.magma__enabled

    String value: ``'True'``, ``'False'``

    Default: ``'False'``

    If ``'True'``, use `magma <http://icl.cs.utk.edu/magma/>`_ for matrix
    computations.

    If ``'False'``, disable magma.

.. attribute:: config.magma__include_path

    Default: ``''``

    Location of the magma headers.

.. attribute:: config.magma__library_path

    Default: ``''``

    Location of the magma library.

.. attribute:: config.ctc__root

    Default: ``''``

    Location of the warp-ctc folder. The folder should contain either a build,
    lib or lib64 subfolder with the shared library (e.g. ``libwarpctc.so``), and another
    subfolder called include, with the CTC library header.

.. attribute:: config.gcc__cxxflags

    Default: ``""``

    Extra parameters to pass to ``gcc`` when compiling.  Extra include paths,
    library paths, configuration options, etc.

.. attribute:: cxx

    Default: Full path to ``g++`` if ``g++`` is present. Empty string otherwise.

    Indicates which C++ compiler to use. If empty, no C++ code is
    compiled.  Aesara automatically detects whether ``g++`` is present and
    disables C++ compilation when it is not.  On Darwin systems (e.g. Mac
    OS X), it looks for ``clang++`` and uses that when available.

    Aesara prints a warning if it detects that no compiler is present.

    This value can point to any compiler binary (full path or not), but things may
    break if the interface is not ``g++``-compatible to some degree.

.. attribute:: config.optimizer_excluding

    Default: ``""``

    A list of rewriter tags that shouldn't be included in the default ``Mode``.
    If multiple tags are provided, separate them by ``':'``.
    For example, to remove the ``Elemwise`` in-place rewrites,
    use the flags: ``optimizer_excluding:inplace_opt``, where
    ``inplace_opt`` is the name of the rewrite group.

    This flag's value cannot be modified during the program execution.

.. attribute:: optimizer_including

    Default: ``""``

    A list of rewriter tags to be included in the default ``Mode``.
    If multiple tags are provided, separate them by ``':'``.

    This flag's value cannot be modified during the program execution.

.. attribute:: optimizer_requiring

    Default: ``""``

    A list of rewriter tags that are required for rewriting in the default
    ``Mode``.
    If multiple tags are provided, separate them by ``':'``.

    This flag's value cannot be modified during the program execution.

.. attribute:: optimizer_verbose

    Bool value: either ``True`` or ``False``

    Default: ``False``

    When ``True``, print the rewrites applied to stdout.

.. attribute:: nocleanup

    Bool value: either ``True`` or ``False``

    Default: ``False``

    If ``False``, source code files are removed when they are no longer needed.
    This causes files for which compilation failed to be deleted.
    Set to ``True`` to keep files for debugging.

.. attribute:: compile

    This section contains attributes which influence the compilation of
    C code for :class:`Op`\s.  Due to historical reasons many attributes outside
    of this section also have an influence over compilation, most
    notably ``cxx``.

.. attribute:: config.compile__timeout

    Positive int value, default: :attr:`compile__wait` * 24

    Time to wait before an un-refreshed lock is broken and stolen (in seconds).
    This is in place to avoid manual cleanup of locks in case a process crashed
    and left a lock in place.

    The refresh time is automatically set to half the timeout value.

.. attribute:: config.compile__wait

    Positive int value, default: 5

    Time to wait between attempts at grabbing the lock if the first
    attempt is not successful (in seconds). The actual time will be between
    :attr:`compile__wait` and :attr:`compile__wait` * 2 to avoid a
    crowding effect on the lock.

.. attribute:: DebugMode

    This section contains various attributes configuring the behaviour of
    :class:`~debugmode.DebugMode`.

.. attribute:: config.DebugMode__check_preallocated_output

    Default: ``''``

    A list of kinds of preallocated memory to use as output buffers for
    each :class:`Op`'s computations, separated by ``:``. Implemented modes are:

    * ``"initial"``: initial storage present in storage map
      (for instance, it can happen in the inner function of :class:`Scan`),
    * ``"previous"``: reuse previously-returned memory,
    * ``"c_contiguous"``: newly-allocated C-contiguous memory,
    * ``"f_contiguous"``: newly-allocated Fortran-contiguous memory,
    * ``"strided"``: non-contiguous memory with various stride patterns,
    * ``"wrong_size"``: memory with bigger or smaller dimensions,
    * ``"ALL"``: placeholder for all of the above.

    In order not to test with preallocated memory, use an empty string, ``""``.

.. attribute:: config.DebugMode__check_preallocated_output_ndim

    Positive int value, default: 4.

    When testing with "strided" preallocated output memory, test
    all combinations of strides over that number of (inner-most)
    dimensions. You may want to reduce that number to reduce memory or
    time usage, but it is advised to keep a minimum of 2.

.. attribute:: config.DebugMode__warn_input_not_reused

    Bool value, default: ``True``

    Generate a warning when a ``destroy_map`` or ``view_map`` says that an
    :class:`Op` will work inplace, but the :class:`Op` does not reuse the input for its
    output.

.. attribute:: config.NanGuardMode__nan_is_error

    Bool value, default: ``True``

    Controls whether ``NanGuardMode`` generates an error when it sees a ``nan``.

.. attribute:: config.NanGuardMode__inf_is_error

    Bool value, default: ``True``

    Controls whether ``NanGuardMode`` generates an error when it sees an ``inf``.

.. attribute:: config.NanGuardMode__big_is_error

    Bool value, default: ``True``

    Controls whether ``NanGuardMode`` generates an error when it sees a
    big value (i.e. a value greater than ``1e10``).

.. attribute:: compute_test_value

    String Value: ``'off'``, ``'ignore'``, ``'warn'``, ``'raise'``.

    Default: ``'off'``

    Setting this attribute to something other than ``'off'`` activates a
    debugging mechanism, for which Aesara executes the graph on-the-fly, as it
    is being built. This allows the user to spot errors early on (such as
    dimension mis-matches) **before** rewrites are applied.

    Aesara will execute the graph using constants and/or shared variables
    provided by the user. Purely symbolic variables (e.g. ``x =
    aesara.tensor.dmatrix()``) can be augmented with test values, by writing to
    their ``.tag.test_value`` attributes (e.g. ``x.tag.test_value = np.ones((5, 4))``).

    When not ``'off'``, the value of this option dictates what happens when
    an :class:`Op`'s inputs do not provide appropriate test values:

        - ``'ignore'`` will do nothing
        - ``'warn'`` will raise a ``UserWarning``
        - ``'raise'`` will raise an exception

.. attribute:: compute_test_value_opt

    As ``compute_test_value``, but it is the value used during Aesara's
    rewriting phase.  This is used to help debug shape errors in Aesara's
    rewrites.

.. attribute:: print_test_value

    Bool value, default: ``False``

    If ``'True'``, Aesara will include the test values in a variable's
    ``__str__`` output.

.. attribute:: exception_verbosity

    String Value: ``'low'``, ``'high'``.

    Default: ``'low'``

    If ``'low'``, the text of exceptions will generally refer to apply nodes
    with short names such as ``'Elemwise{add_no_inplace}'``. If ``'high'``,
    some exceptions will also refer to :class:`Apply` nodes with long descriptions
    like:

    ::

        A. Elemwise{add_no_inplace}
              B. log_likelihood_v_given_h
              C. log_likelihood_h


.. attribute:: config.cmodule__warn_no_version

    Bool value, default: ``False``

    If ``True``, will print a warning when compiling one or more :class:`Op` with C
    code that can't be cached because there is no ``c_code_cache_version()``
    function associated to at least one of those :class:`Op`\s.

.. attribute:: config.cmodule__remove_gxx_opt

    Bool value, default: ``False``

    If ``True``, Aesara will remove the ``-O*`` parameter passed to ``g++``.
    This is useful for debugging objects compiled by Aesara.  The parameter
    ``-g`` is also passed by default to ``g++``.

.. attribute:: config.cmodule__compilation_warning

    Bool value, default: ``False``

    If ``True``, Aesara will print compilation warnings.

.. attribute:: config.cmodule__preload_cache

    Bool value, default: ``False``

    If set to ``True``, Aesara will preload the C module cache at import time

.. attribute:: config.cmodule__age_thresh_use

    Int value, default: ``60 * 60 * 24 * 24``  # 24 days

    The time after which a compiled C module won't be reused by Aesara (in
    seconds). C modules are automatically deleted 7 days after that time.

.. attribute:: config.cmodule__debug

    Bool value, default: ``False``

    If ``True``, define a DEBUG macro (if one doesn't already exist) for all
    compiled C code.

.. attribute:: config.traceback__limit

    Int value, default: 8

    The number of traceback stack levels to keep for Aesara variable
    definitions.

.. attribute:: config.traceback__compile_limit

    Bool value, default: 0

    The number of traceback stack levels to keep for variables during Aesara
    compilation. When this value is greater than zero, it will make Aesara keep
    internal stack traces.

.. attribute:: config.metaopt__verbose

    Int value, default: 0

    The verbosity level of the meta-rewriter: ``0`` for silent, ``1`` to only
    warn when Aesara cannot meta-rewrite an :class:`Op`, ``2`` for full output (e.g.
    timings and the rewrites selected).


.. attribute:: config.metaopt__optimizer_excluding

    Default: ``""``

    A list of rewrite tags that we don't want included in the meta-rewriter.
    Multiple tags are separate by ``':'``.

.. attribute:: config.metaopt__optimizer_including

    Default: ``""``

    A list of rewriter tags to be included during meta-rewriting.
    Multiple tags are separate by ``':'``.

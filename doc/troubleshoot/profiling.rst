
.. _tut_profiling:

=========================
Profiling Aesara function
=========================

.. note::

    This method replaces the old `ProfileMode`. Do not use `ProfileMode`
    anymore.

Besides checking for errors, another important task is to profile your code in
terms of speed and/or memory usage.  You can profile your functions using either
of the following two options:

1. Use the Aesara flag :attr:`config.profile` to enable profiling.
    - To enable the memory profiler use the Aesara flag:
      :attr:`config.profile_memory` in addition to :attr:`config.profile`.
    - Moreover, to enable the profiling of Aesara rewrite phases,
      use the Aesara flag: :attr:`config.profile_optimizer` in addition
      to :attr:`config.profile`.
    - You can also use the Aesara flags :attr:`profiling__n_apply`,
      :attr:`profiling__n_ops` and :attr:`profiling__min_memory_size`
      to modify the quantity of information printed.

2. Pass the argument :attr:`profile=True` to the function :func:`aesara.function
   <function.function>` and then call :attr:`f.profile.summary()` for a single
   function.
    - Use this option when you want to profile not all the
      functions but only one or more specific function(s).
    - You can also combine the profile results of many functions:

      .. doctest::
          :hide:

          profile = aesara.compile.ProfileStats()
          f = aesara.function(..., profile=profile)  # doctest: +SKIP
          g = aesara.function(..., profile=profile)  # doctest: +SKIP
          ...  # doctest: +SKIP
          profile.summary()



The profiler will output one profile per Aesara function and produce a profile
that is the sum of the printed profiles. Each profile contains four sections:
global info, class info, `Op`\s info and `Apply` node info.

In the global section, the "Message" is the name of the Aesara
function. :func:`aesara.function` has an optional parameter ``name`` that
defaults to ``None``. Change it to something else to help profile many
Aesara functions. In that section, we also see the number of times the
function was called (1) and the total time spent in all those
calls. The time spent in :meth:`Function.vm.__call__` and in thunks is useful
to understand Aesara's overhead.

Also, we see the time spent in the two parts of the compilation process:
rewriting (i.e. modifying the graph to make it more stable/faster) and the
linking (i.e. compile C code and make the Python callable returned by
:func:`aesara.function`).

The class, `Op`\s and `Apply` nodes sections have the same information: i.e.
information about the `Apply` nodes that ran. The `Op`\s section takes the
information from the `Apply` section and merges it with the `Apply` nodes that have
exactly the same `Op`. If two `Apply` nodes in the graph have two `Op`\s that
compare equal, they will be merged. Some `Op`\s, like `Elemwise`, will not
compare equal if their parameters differ, so the class section will merge more
`Apply` nodes than the `Op`\s section.

Note that the profile also shows which `Op`\s were run with C
implementation.

Developers wishing to optimize the performance of their graph should
focus on the worst offending `Op`\s and `Apply` nodes--either by optimizing
an implementation, providing a missing C implementation, or by writing
a graph rewrite that eliminates the offending `Op` altogether.

Here is some example output when Aesara's rewrites are disabled. With all
rewrites enabled, there would be only one `Op` remaining in the graph.

To run the example:

    AESARA_FLAGS=optimizer_excluding=fusion:inplace,profile=True python doc/tutorial/profiling_example.py

The output:

.. literalinclude:: profiling_example_out.prof

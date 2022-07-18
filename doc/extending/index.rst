
.. _extending:

================
Extending Aesara
================

This advanced tutorial is for users who want to extend Aesara with new :class:`Type`\s,
new operations (i.e. :class:`Op`\s), and new graph rewrites. This first page of the
tutorial mainly focuses on the Python implementation of an :class:`Op` and then
proposes an overview of the most important methods that define an :class:`Op`.
The second page of the tutorial (:ref:`creating_a_c_op`) provides then
information on the C implementation of an :class:`Op`. The rest of the tutorial
goes more in depth on advanced topics related to :class:`Op`\s, such as how to write
efficient code for an :class:`Op` and how to write an rewrite to speed up the
execution of an :class:`Op`.

Along the way, this tutorial also introduces many aspects of how Aesara works,
so it is also good for you if you are interested in getting more under the hood
with Aesara itself.

.. note::

    Before tackling this more advanced presentation, it is highly recommended
    to read the introductory :ref:`Tutorial<tutorial>`, especially the sections
    that introduce the Aesara graphs, as providing a novel Aesara :class:`Op` requires a
    basic understanting of the Aesara graphs.

    See also the :ref:`dev_start_guide` for information regarding the
    versioning framework, namely about Git and GitHub, regarding the
    development workflow and how to make a quality contribution.

.. toctree::

    graphstructures
    graph_rewriting
    op
    creating_an_op
    creating_a_c_op
    creating_a_numba_jax_op
    pipeline
    type
    ctype
    inplace
    scan
    other_ops
    using_params
    unittest
    extending_faq
    tips

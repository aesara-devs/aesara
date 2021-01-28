
.. _extending:

================
Extending Aesara
================

This advanced tutorial is for users who want to extend Aesara with new Types,
new Operations (Ops), and new graph optimizations. This first page of the
tutorial mainly focuses on the Python implementation of an Op and then
proposes an overview of the most important methods that define an op.
The second page of the tutorial (:ref:`extending_aesara_c`) provides then
information on the C implementation of an Op. The rest of the tutorial
goes more in depth on advanced topics related to Ops, such as how to write
efficient code for an Op and how to write an optimization to speed up the
execution of an Op.

Along the way, this tutorial also introduces many aspects of how Aesara works,
so it is also good for you if you are interested in getting more under the hood
with Aesara itself.

.. note::

    Before tackling this more advanced presentation, it is highly recommended
    to read the introductory :ref:`Tutorial<tutorial>`, especially the sections
    that introduce the Aesara Graphs, as providing a novel Aesara op requires a
    basic understanting of the Aesara Graphs.

    See also the :ref:`dev_start_guide` for information regarding the
    versioning framework, namely about *git* and *GitHub*, regarding the
    development workflow and how to make a quality contribution.

.. toctree::

    extending_aesara
    extending_aesara_c
    fibby
    pipeline
    aesara_vs_c
    graphstructures
    type
    op
    inplace
    other_ops
    ctype
    cop
    using_params
    extending_aesara_gpu
    optimization
    tips
    unittest
    scan
    extending_faq

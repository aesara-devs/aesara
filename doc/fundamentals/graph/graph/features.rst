.. _libdoc_graph_features:

================================================
:mod:`features` -- [doc TODO]
================================================

.. module:: aesara.graph.features
   :platform: Unix, Windows
   :synopsis: Aesara Internals
.. moduleauthor:: LISA

Guide
=====

.. class:: Bookkeeper(object)

.. class:: History(object)

    .. method:: revert(fgraph, checkpoint)
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements).  A checkpoint at any
        given time can be obtained using self.checkpoint().

.. class:: Validator(object)

.. class:: ReplaceValidate(History, Validator)

    .. method:: replace_validate(fgraph, var, new_var, reason=None)

.. class:: NodeFinder(Bookkeeper)

.. class:: PrintListener(object)

.. _libdoc_tensor_nnet_ctc:

==================================================================================
:mod:`aesara.tensor.nnet.ctc` -- Connectionist Temporal Classification (CTC) loss
==================================================================================

.. note::

    Usage of connectionist temporal classification (CTC) loss Op, requires that
    the `warp-ctc <https://github.com/baidu-research/warp-ctc>`_ library is
    available. In case the warp-ctc library is not in your compiler's library path,
    the ``config.ctc__root`` configuration option must be appropriately set to the
    directory containing the warp-ctc library files.

.. note::

   This interface is the preferred interface.

.. note::

    Unfortunately, Windows platforms are not yet supported by the underlying
    library.

.. module:: aesara.tensor.nnet.ctc
   :platform: Unix
   :synopsis: Connectionist temporal classification (CTC) loss Op, using the warp-ctc library
.. moduleauthor:: `João Victor Risso <https://github.com/joaovictortr>`_

.. autofunction:: aesara.tensor.nnet.ctc.ctc
.. autoclass:: aesara.tensor.nnet.ctc.ConnectionistTemporalClassification

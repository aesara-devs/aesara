.. _libdoc_gpuarray_ctc:

================================================================================
:mod:`aesara.gpuarray.ctc` -- Connectionist Temporal Classification (CTC) loss
================================================================================


.. warning::

    This is not the recomanded user interface. Use :ref:`the CPU
    interface <libdoc_tensor_nnet_ctc>`. It will get moved
    automatically to the GPU.

.. note::

    Usage of connectionist temporal classification (CTC) loss Op, requires that
    the `warp-ctc <https://github.com/baidu-research/warp-ctc>`_ library is
    available. In case the warp-ctc library is not in your compiler's library path,
    the :attr:`config.ctc__root` configuration option must be appropriately set to the
    directory containing the warp-ctc library files.

.. note::

    Unfortunately, Windows platforms are not yet supported by the underlying
    library.

.. module:: aesara.gpuarray.ctc
   :platform: Unix
   :synopsis: Connectionist temporal classification (CTC) loss Op, using the warp-ctc library
.. moduleauthor:: `João Victor Risso <https://github.com/joaovictortr>`_

.. autofunction:: aesara.gpuarray.ctc.gpu_ctc
.. autoclass:: aesara.gpuarray.ctc.GpuConnectionistTemporalClassification

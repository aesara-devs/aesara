.. _libdoc_tensor_nnet_conv:

==========================================================
:mod:`conv` -- Ops for convolutional neural nets
==========================================================

.. note::

    Two similar implementation exists for conv2d:

        :func:`signal.conv2d <aesara.tensor.signal.conv.conv2d>` and
        :func:`nnet.conv2d <aesara.tensor.nnet.conv2d>`.

    The former implements a traditional
    2D convolution, while the latter implements the convolutional layers
    present in convolutional neural networks (where filters are 3D and pool
    over several input channels).

.. module:: conv
   :platform: Unix, Windows
   :synopsis: ops for signal processing
.. moduleauthor:: LISA


The recommended user interface are:

- :func:`aesara.tensor.nnet.conv2d` for 2d convolution
- :func:`aesara.tensor.nnet.conv3d` for 3d convolution

With those new interface, Aesara will automatically use the fastest
implementation in many cases. On the CPU, the implementation is a GEMM
based one. On the GPU, there is a GEMM based and :ref:`cuDNN
<libdoc_gpuarray_dnn>` version.

By default on the GPU, if cuDNN is available, it will be used,
otherwise we will fall back to using gemm based version (slower than
cuDNN in most cases and uses more memory). To get an error if cuDNN
can not be used, you can supply the Aesara flag ``dnn.enable=True``.

Either cuDNN and the gemm version can be disabled using the Aesara flags
``optimizer_excluding=conv_dnn`` and ``optimizer_excluding=conv_gemm``,
respectively. If both are disabled, it will raise an error.


For the cuDNN version, there are different algorithms with different
memory/speed trade-offs. Manual selection of the right one is very
difficult as it depends on the shapes and hardware. So it can change
for each layer. An auto-tuning mode exists and can be activated by
those flags: ``dnn__conv__algo_fwd=time_once``,
``dnn__conv__algo_bwd_data=time_once`` and
``dnn__conv__algo_bwd_filter=time_once``. Note, they are good mostly
when the shape do not change.

This auto-tuning has the inconvenience that the first call is much
slower as it tries and times each implementation it has. So if you
benchmark, it is important that you remove the first call from your
timing.

Also, a meta-optimizer has been implemented for the gpu convolution
implementations to automatically choose the fastest implementation
for each specific convolution in your graph. For each instance, it will
compile and benchmark each applicable implementation and choose the
fastest one. It can be enabled using ``optimizer_including=conv_meta``.
The meta-optimizer can also selectively disable cudnn and gemm version
using the Aesara flag ``metaopt__optimizer_excluding=conv_dnn`` and
``metaopt__optimizer_excluding=conv_gemm`` respectively.


.. note::

    Aesara had older user interface like
    aesara.tensor.nnet.conv.conv2d. Do not use them anymore. They
    will give you slower code and won't allow easy switch between CPU
    and GPU computation. They also support less type of convolution.


Implementation Details
======================

This section gives more implementation detail. Most of the time you do
not need to read it. Aesara will select it for you.


- Implemented operators for neural network 2D / image convolution:
    - :func:`nnet.conv.conv2d <aesara.tensor.nnet.conv.conv2d>`.
      old 2d convolution. DO NOT USE ANYMORE.

    - :func:`GpuCorrMM <aesara.gpuarray.blas.GpuCorrMM>`
      This is a GPU-only 2d correlation implementation taken from
      `caffe's CUDA implementation <https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu>`_. It does not flip the kernel.

      For each element in a batch, it first creates a
      `Toeplitz <http://en.wikipedia.org/wiki/Toeplitz_matrix>`_ matrix in a CUDA kernel.
      Then, it performs a ``gemm`` call to multiply this Toeplitz matrix and the filters
      (hence the name: MM is for matrix multiplication).
      It needs extra memory for the Toeplitz matrix, which is a 2D matrix of shape
      ``(no of channels * filter width * filter height, output width * output height)``.

    - :func:`CorrMM <aesara.tensor.nnet.corr.CorrMM>`
      This is a CPU-only 2d correlation implementation taken from
      `caffe's cpp implementation <https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cpp>`_.
      It does not flip the kernel.
    - :func:`dnn_conv <aesara.gpuarray.dnn.dnn_conv>` GPU-only
      convolution using NVIDIA's cuDNN library.

- Implemented operators for neural network 3D / video convolution:
    - :func:`GpuCorr3dMM <aesara.gpuarray.blas.GpuCorr3dMM>`
      This is a GPU-only 3d correlation relying on a Toeplitz matrix
      and gemm implementation (see :func:`GpuCorrMM <aesara.sandbox.cuda.blas.GpuCorrMM>`)
      It needs extra memory for the Toeplitz matrix, which is a 2D matrix of shape
      ``(no of channels * filter width * filter height * filter depth, output width * output height * output depth)``.
    - :func:`Corr3dMM <aesara.tensor.nnet.corr3d.Corr3dMM>`
      This is a CPU-only 3d correlation implementation based on
      the 2d version (:func:`CorrMM <aesara.tensor.nnet.corr.CorrMM>`).
      It does not flip the kernel. As it provides a gradient, you can use it as a
      replacement for nnet.conv3d. For convolutions done on CPU,
      nnet.conv3d will be replaced by Corr3dMM.

    - :func:`dnn_conv3d <aesara.gpuarray.dnn.dnn_conv3d>` GPU-only
      3D convolution using NVIDIA's cuDNN library (as :func:`dnn_conv <aesara.gpuarray.dnn.dnn_conv>` but for 3d).

      If cuDNN is available, by default, Aesara will replace all nnet.conv3d
      operations with dnn_conv.

    - :func:`conv3d2d <aesara.tensor.nnet.conv3d2d.conv3d>`
      Another conv3d implementation that uses the conv2d with data reshaping.
      It is faster in some corner cases than conv3d. It flips the kernel.

.. autofunction:: aesara.tensor.nnet.conv2d
.. autofunction:: aesara.tensor.nnet.conv2d_transpose
.. autofunction:: aesara.tensor.nnet.conv3d
.. autofunction:: aesara.tensor.nnet.conv3d2d.conv3d
.. autofunction:: aesara.tensor.nnet.conv.conv2d

.. automodule:: aesara.tensor.nnet.abstract_conv
    :members:

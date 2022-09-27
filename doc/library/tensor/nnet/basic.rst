.. _libdoc_tensor_nnet_basic:

======================================================
:mod:`basic` -- Basic Ops for neural networks
======================================================

.. module:: aesara.tensor.nnet.basic
   :platform: Unix, Windows
   :synopsis: Ops for neural networks
.. moduleauthor:: LISA


- Sigmoid
   - :func:`sigmoid`
   - :func:`ultra_fast_sigmoid`
   - :func:`hard_sigmoid`

- Others
   - :func:`softplus`
   - :func:`softmax`
   - :func:`softsign`
   - :func:`relu() <aesara.tensor.nnet.relu>`
   - :func:`elu() <aesara.tensor.nnet.elu>`
   - :func:`selu() <aesara.tensor.nnet.selu>`
   - :func:`binary_crossentropy`
   - :func:`sigmoid_binary_crossentropy`
   - :func:`.categorical_crossentropy`
   - :func:`h_softmax() <aesara.tensor.nnet.h_softmax>`
   - :func:`confusion_matrix <aesara.tensor.nnet.confusion_matrix>`

.. function:: sigmoid(x)

   Returns the standard sigmoid nonlinearity applied to x
    :Parameters: *x* - symbolic Tensor (or compatible)
    :Return type: same as x
    :Returns: element-wise sigmoid: :math:`sigmoid(x) = \frac{1}{1 + \exp(-x)}`.
    :note: see :func:`ultra_fast_sigmoid` or :func:`hard_sigmoid` for faster versions.
        Speed comparison for 100M float64 elements on a Core2 Duo @ 3.16 GHz:

          - hard_sigmoid: 1.0s
          - ultra_fast_sigmoid: 1.3s
          - sigmoid (with amdlibm): 2.3s
          - sigmoid (without amdlibm): 3.7s

        Precision: sigmoid(with or without amdlibm) > ultra_fast_sigmoid > hard_sigmoid.

   .. image:: sigmoid_prec.png

   Example:

   .. testcode::

       import aesara.tensor as at

       x, y, b = at.dvectors('x', 'y', 'b')
       W = at.dmatrix('W')
       y = at.sigmoid(at.dot(W, x) + b)

   .. note:: The underlying code will return an exact 0 or 1 if an
      element of x is too small or too big.

.. function:: ultra_fast_sigmoid(x)

   Returns an approximate standard :func:`sigmoid` nonlinearity applied to ``x``.
    :Parameters: ``x`` - symbolic Tensor (or compatible)
    :Return type: same as ``x``
    :Returns: approximated element-wise sigmoid: :math:`sigmoid(x) = \frac{1}{1 + \exp(-x)}`.
    :note: To automatically change all :func:`sigmoid`\ :class:`Op`\s to this version, use
      the Aesara rewrite `local_ultra_fast_sigmoid`. This can be done
      with the Aesara flag ``optimizer_including=local_ultra_fast_sigmoid``.
      This rewrite is done late, so it should not affect stabilization rewrites.

   .. note:: The underlying code will return 0.00247262315663 as the
       minimum value and 0.997527376843 as the maximum value. So it
       never returns 0 or 1.

   .. note:: Using directly the `ultra_fast_sigmoid` in the graph will
       disable stabilization rewrites associated with it. But
       using the rewrite to insert them won't disable the
       stability rewrites.


.. function:: hard_sigmoid(x)

   Returns an approximate standard :func:`sigmoid` nonlinearity applied to `1x1`.
    :Parameters: ``x`` - symbolic Tensor (or compatible)
    :Return type: same as ``x``
    :Returns: approximated element-wise sigmoid: :math:`sigmoid(x) = \frac{1}{1 + \exp(-x)}`.
    :note: To automatically change all :func:`sigmoid`\ :class:`Op`\s to this version, use
      the Aesara rewrite `local_hard_sigmoid`. This can be done
      with the Aesara flag ``optimizer_including=local_hard_sigmoid``.
      This rewrite is done late, so it should not affect
      stabilization rewrites.

   .. note:: The underlying code will return an exact 0 or 1 if an
      element of ``x`` is too small or too big.

   .. note:: Using directly the `ultra_fast_sigmoid` in the graph will
       disable stabilization rewrites associated with it. But
       using the rewrites to insert them won't disable the
       stability rewrites.

.. function:: softplus(x)

   Returns the softplus nonlinearity applied to x
    :Parameter: *x* - symbolic Tensor (or compatible)
    :Return type: same as x
    :Returns: element-wise softplus: :math:`softplus(x) = \log_e{\left(1 + \exp(x)\right)}`.

   .. note:: The underlying code will return an exact 0 if an element of x is too small.

   .. testcode::

       x, y, b = at.dvectors('x', 'y', 'b')
       W = at.dmatrix('W')
       y = at.nnet.softplus(at.dot(W,x) + b)

.. function:: softsign(x)

   Return the elemwise softsign activation function
   :math:`\\varphi(\\mathbf{x}) = \\frac{1}{1+|x|}`


.. function:: softmax(x)

   Returns the softmax function of x:
    :Parameter: *x* symbolic **2D** Tensor (or compatible).
    :Return type: same as x
    :Returns: a symbolic 2D tensor whose ijth element is  :math:`softmax_{ij}(x) = \frac{\exp{x_{ij}}}{\sum_k\exp(x_{ik})}`.

   The softmax function will, when applied to a matrix, compute the softmax values row-wise.

    :note: this supports hessian free as well.  The code of
       the softmax op is more numerically stable because it uses this code:

       .. code-block:: python

           e_x = exp(x - x.max(axis=1, keepdims=True))
           out = e_x / e_x.sum(axis=1, keepdims=True)

   Example of use:

   .. testcode::

       x, y, b = at.dvectors('x', 'y', 'b')
       W = at.dmatrix('W')
       y = at.nnet.softmax(at.dot(W,x) + b)

.. autofunction:: aesara.tensor.nnet.relu

.. autofunction:: aesara.tensor.nnet.elu

.. autofunction:: aesara.tensor.nnet.selu

.. function:: binary_crossentropy(output,target)

   Computes the binary cross-entropy between a target and an output:
    :Parameters:

       * *target* - symbolic Tensor (or compatible)
       * *output* - symbolic Tensor (or compatible)

    :Return type: same as target
    :Returns: a symbolic tensor, where the following is applied element-wise :math:`crossentropy(t,o) = -(t\cdot log(o) + (1 - t) \cdot log(1 - o))`.

   The following block implements a simple auto-associator with a
   sigmoid nonlinearity and a reconstruction error which corresponds
   to the binary cross-entropy (note that this assumes that x will
   contain values between 0 and 1):

   .. testcode::

       x, y, b, c = at.dvectors('x', 'y', 'b', 'c')
       W = at.dmatrix('W')
       V = at.dmatrix('V')
       h = at.sigmoid(at.dot(W, x) + b)
       x_recons = at.sigmoid(at.dot(V, h) + c)
       recon_cost = at.nnet.binary_crossentropy(x_recons, x).mean()

.. function:: sigmoid_binary_crossentropy(output,target)

   Computes the binary cross-entropy between a target and the sigmoid of an output:
    :Parameters:

       * *target* - symbolic Tensor (or compatible)
       * *output* - symbolic Tensor (or compatible)

    :Return type: same as target
    :Returns: a symbolic tensor, where the following is applied element-wise :math:`crossentropy(o,t) = -(t\cdot log(sigmoid(o)) + (1 - t) \cdot log(1 - sigmoid(o)))`.

   It is equivalent to `binary_crossentropy(sigmoid(output), target)`,
   but with more efficient and numerically stable computation, especially when
   taking gradients.

   The following block implements a simple auto-associator with a
   sigmoid nonlinearity and a reconstruction error which corresponds
   to the binary cross-entropy (note that this assumes that x will
   contain values between 0 and 1):

   .. testcode::

       x, y, b, c = at.dvectors('x', 'y', 'b', 'c')
       W = at.dmatrix('W')
       V = at.dmatrix('V')
       h = at.sigmoid(at.dot(W, x) + b)
       x_precons = at.dot(V, h) + c
       # final reconstructions are given by sigmoid(x_precons), but we leave
       # them unnormalized as sigmoid_binary_crossentropy applies sigmoid
       recon_cost = at.sigmoid_binary_crossentropy(x_precons, x).mean()

.. function:: categorical_crossentropy(coding_dist,true_dist)

    Return the cross-entropy between an approximating distribution and a true distribution.
    The cross entropy between two probability distributions measures the average number of bits
    needed to identify an event from a set of possibilities, if a coding scheme is used based
    on a given probability distribution q, rather than the "true" distribution p. Mathematically, this
    function computes :math:`H(p,q) = - \sum_x p(x) \log(q(x))`, where
    p=true_dist and q=coding_dist.

    :Parameters:

       * *coding_dist* - symbolic 2D Tensor (or compatible). Each row
         represents a distribution.
       * *true_dist* - symbolic 2D Tensor **OR** symbolic vector of ints.  In
         the case of an integer vector argument, each element represents the
         position of the '1' in a 1-of-N encoding (aka "one-hot" encoding)

    :Return type: tensor of rank one-less-than `coding_dist`

   .. note:: An application of the scenario where *true_dist* has a
       1-of-N representation is in classification with softmax
       outputs. If `coding_dist` is the output of the softmax and
       `true_dist` is a vector of correct labels, then the function
       will compute ``y_i = - \log(coding_dist[i, one_of_n[i]])``,
       which corresponds to computing the neg-log-probability of the
       correct class (which is typically the training criterion in
       classification settings).

   .. testsetup::

      import aesara
      o = aesara.tensor.ivector()

   .. testcode::

       y = at.nnet.softmax(at.dot(W, x) + b)
       cost = at.nnet.categorical_crossentropy(y, o)
       # o is either the above-mentioned 1-of-N vector or 2D tensor


.. autofunction:: aesara.tensor.nnet.h_softmax

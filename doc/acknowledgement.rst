.. _acknowledgement:


Acknowledgements
================

.. note:

   This page is in construction. We are missing sources.


* The developers of `NumPy <http://numpy.scipy.org/>`_. Theano is based on its ndarray object and uses much of its implementation.
* The developers of `SciPy <http://scipy.org/>`_. Our sparse matrix support uses their sparse matrix objects. We also reuse other parts.
* The developers of `Theano <https://github.com/Theano/Theano>`_
* All `Aesara contributors <https://github.com/aesara-devs/aesara/graphs/contributors>`_.
* All Theano users that have given us feedback.
* Our random number generator implementation on CPU and GPU uses the MRG31k3p algorithm that is described in:

    P. L'Ecuyer and R. Touzin, `Fast Combined Multiple Recursive Generators with Multipliers of the form a = +/- 2^d +/- 2^e <http://www.informs-sim.org/wsc00papers/090.PDF>`_, Proceedings of the 2000 Winter Simulation Conference, Dec. 2000, 683--689.

  We were authorized by Pierre L'Ecuyer to copy/modify his Java implementation in the `SSJ <http://www.iro.umontreal.ca/~simardr/ssj/>`_ software and to relicense it under BSD 3-Clauses in Theano.

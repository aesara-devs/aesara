.. _reference_tensor_mathematical_functions:
.. currentmodule:: aesara.tensor


Mathematical functions
======================

Trigonometric functions
-----------------------

.. autosummary::
   :toctree: _autosummary

   sin
   cos
   tan
   arcsin
   arccos
   arctan
   arctan2
   deg2rad
   rad2deg

Hyperbolic functions
--------------------

.. autosummary::
   :toctree: _autosummary

   sinh
   cosh
   tanh
   arcsinh
   arccosh
   arctanh

Rounding
--------

.. autosummary::
   :toctree: _autosummary

   round
   iround
   floor
   ceil
   trunc
   round_half_to_even
   round_half_away_from_zero

Sums, products, differences
---------------------------

.. autosummary::
   :toctree: _autosummary

   sum
   cumsum
   prod
   cumprod
   diff

Exponents and logarithms
------------------------

.. autosummary::
   :toctree: _autosummary

   exp
   expm1
   exp2
   log
   log2
   log10
   log1p
   logaddexp
   square
   sqrt
   power

Special functions
-----------------

.. autosummary::
   :toctree: _autosummary

   erfc
   erfcx
   erfcinv
   erfinv
   owens_t
   gamma
   gammaln
   psi
   tri_gamma
   chi2sf
   gammainc
   gammaincc
   gammau
   hyp2f1
   hyp2f1_der
   j0
   j1
   jv
   i0
   i1
   iv
   sigmoid
   softplus
   special.softmax
   special.log_softmax
   extra_ops.to_one_hot
   special.poch
   special.factorial
   log1mexp
   betainc
   logsumexp
   xlogx.xlogx
   xlogx.xlogy0

Arithmetic operations
---------------------

.. autosummary::
   :toctree: _autosummary

   add
   reciprocal
   power
   mod
   abs
   neg
   pow
   sgn
   mul
   sub
   int_div
   ceil_intdiv
   true_div
   divmod


.. doctest::
   :options: +SKIP

   >>> a, b = at.itensor3(), at.itensor3() # example inputs
   >>> a + 3      # at.add(a, 3) -> itensor3
   >>> 3 - a      # at.sub(3, a)
   >>> a * 3.5    # at.mul(a, 3.5) -> ftensor3 or dtensor3 (depending on casting)
   >>> 2.2 / a    # at.truediv(2.2, a)
   >>> 2.2 // a   # at.intdiv(2.2, a)
   >>> 2.2**a     # at.pow(2.2, a)
   >>> b % a      # at.mod(b, a)

Extrema
-------

.. autosummary::
   :toctree: _autosummary

   maximum
   minimum
   topk
   argtopk
   topk_and_argtopk

Complex numbers
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   angle
   real
   imag
   conjugate
   complex_from_polar
   complex

Miscellaneous
-------------

.. autosummary::
   :toctree: _autosummary

   clip

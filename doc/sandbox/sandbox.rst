Basically, this file contains stuff that should be documented, but is not.

Feel free to contribute things that you want documented, as well as to add
or correct documentation.


======================================
How do you define the grad function?
======================================

Let's talk about defining the :meth:`Op.grad` function in an :class:`Op`, using an
illustrative example.

In Poisson regression (Ranzato and Szummer, 2008), the target *t* is
integer data, which we predict using a continuous output *o*.
In the negative log likelihood of the Poisson regressor, there is a term:

.. math::

    \log(t!)

Let's say we write a logfactorial :class:`Op`. We then compute the gradient

You should define gradient, even if it is undefined.
[give log factorial example]

If an :class:`Op` does not define ``grad``, but this :class:`Op` does not appear in the path when
you compute the gradient, then there is no problem.

If an :class:`Op` does not define ``grad``, and this :class:`Op` *does* appear in the path when
you compute the gradient, **WRITEME**.

Gradients for a particular variable can be one of four kinds:
1) forgot to implement it

You will get an exception of the following form::

    aesara.graph.utils.MethodNotDefined: ('grad', <class 'pylearn.algorithms.sandbox.cost.LogFactorial'>, 'LogFactorial')

2) a symbolic variable
3) None / zero
4) undefined mathematically

currently, there is no way for a ``grad()`` method to distinguish between cases 3
and 4
but the distinction is important because graphs with type-3 gradients are ok
to run, whereas graphs with type-4 gradients are not.
so I suggested that Joseph return a type-4 gradient by defining an :class:`Op` with no
perform method.
the idea would be that this would suit the graph-construction phase, but would
prevent linking.
how does that sound to you?

**This documentation is useful when we show users how to write :class:`Op`\s.**

======================================
What is staticmethod, st_impl?
======================================

``st_impl`` is an optional method in an :class:`Op`.
``@staticmethod`` is a Python decorator for a class method that does not
implicitly take the class instance as a first argument. Hence, st_impl
can be used for :class:`Op` implementations when no information from the :class:`Op`
instance is needed. This can be useful for testing an implementation.
See the ``XlogX`` class below for an example.

**This documentation is useful when we show users how to write :class:`Op`\s.
Olivier says this behavior should be discouraged but I feel that st_impl
should be encouraged where possible.**

============================================================
how do we write scalar ops and upgrade them to tensor ops?
============================================================

**Olivier says that** :class:`~aesara.tensor.xlogx.XlogX` **gives a good example. In fact, I would
like to beef xlogx up into our running example for demonstrating how to
write an :class:`Op`:**

.. code-block:: python

    class XlogX(scalar.UnaryScalarOp):
        """
        Compute X * log(X), with special case 0 log(0) = 0.
        """
        @staticmethod
        def st_impl(x):
            if x == 0.0:
                return 0.0
            return x * numpy.log(x)
        def impl(self, x):
            return XlogX.st_impl(x)
        def grad(self, inp, grads):
            x, = inp
            gz, = grads
            return [gz * (1 + scalar.log(x))]
        def c_code(self, node, name, inp, out, sub):
            x, = inp
            z, = out
            if node.inputs[0].type in [scalar.float32, scalar.float64]:
                return """%(z)s =
                    %(x)s == 0.0
                    ? 0.0
                    : %(x)s * log(%(x)s);""" % locals()
            raise NotImplementedError('only floatingpoint is implemented')
    scalar_xlogx  = XlogX(scalar.upgrade_to_float, name='scalar_xlogx')
    xlogx = aesara.tensor.elemwise.Elemwise(scalar_xlogx, name='xlogx')

**It is also necessary to talk about UnaryScalarOp vs. BinaryOp.**

UnaryScalarOp is the same as scalar.ScalarOp with member variable nin=1.
**give an example of this**

=======================================================
How to use the `PrintOp`
=======================================================

** This is also useful in the How to write an :class:`Op` tutorial. **

=======================================================
Mammouth
=======================================================

**This is internal documentation. Guillaume can you make sure to hit these points:**

export AESARA_BLAS_LDFLAGS='-lmkl -liomp5 -fopenmp'

**Do we want the following:**

export OMP_NUM_THREADS=2

=======================================================
Type checking
=======================================================

    * Are there functions for doing type checking?
        like dtype of this matrix is an int-type (not just int32
        or int64)
        "if isinstance(item, int):" is the preferred way to do it in
        python now, so mimic this
        If the type is wrong, what exception should be raised?

======================================
More simple numpy stuff
======================================

    * If we have a matrix with only one row, how do we convert it to a vector?
        ``x.reshape(x.size)``
        You can also use ``resize`` but there is not reason to ''resize''
    * How do you convert the type of a numpy array?
        ``aesara._asarray(x, dtype = 'int32')``
        Note that using ``numpy.asarray`` is potentially dangerous, due to
        a problem in numpy where the type may not be properly set (see
        numpy's Track ticket #870).


=========================================
How to reuse (overwrite) a storage tensor
=========================================

``aesara.compile.io.Out(gw1, borrow = True)`` for that value in
``aesara.compile.function.function``

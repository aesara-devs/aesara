
.. _op_contract:

=============
:class:`Op`\s
=============

An :class:`Op` is a :ref:`graph object <graphstructures>` that defines and performs computations in a graph.

It has to define the following methods.

.. function:: make_node(*inputs)

  This method is responsible for creating output :class:`Variable`\s of a
  suitable symbolic `Type` to serve as the outputs of this :Class:`Op`'s
  application.  The :class:`Variable`\s found in ``*inputs`` must be operated on
  using Aesara's symbolic language to compute the symbolic output
  :class:`Variable`\s. This method should put these outputs into an :class:`Apply`
  instance, and return the :class:`Apply` instance.

  This method creates an :class:`Apply` node representing the application of
  the `Op` on the inputs provided. If the `Op` cannot be applied to these
  inputs, it must raise an appropriate exception.

  The inputs of the :class:`Apply` instance returned by this call must be
  ordered correctly: a subsequent ``self.make_node(*apply.inputs)``
  must produce something equivalent to the first ``apply``.

.. function:: perform(node, inputs, output_storage)

  This method computes the function associated to this :class:`Op`. ``node`` is
  an :class:`Apply` node created by the :class:`Op`'s :meth:`Op.make_node` method. ``inputs``
  is a list of references to data to operate on using non-symbolic
  statements, (i.e., statements in Python, NumPy). ``output_storage``
  is a list of storage cells where the variables of the computation
  must be put.

  More specifically:

    - ``node``: This is a reference to an :class:`Apply` node which was previously
      obtained via the :meth:`Op.make_node` method. It is typically not
      used in simple :class:`Op`\s, but it contains symbolic information that
      could be required for complex :class:`Op`\s.

    - ``inputs``: This is a list of data from which the values stored in ``output_storage``
      are to be computed using non-symbolic language.

    - ``output_storage``: This is a list of storage cells where the output is to be stored.
      A storage cell is a one-element list. It is forbidden to change
      the length of the list(s) contained in ``output_storage``.
      There is one storage cell for each output of the :class:`Op`.

      The data put in ``output_storage`` must match the type of the
      symbolic output. This is a situation where the ``node`` argument
      can come in handy.

      A function :class:`Mode` may allow ``output_storage`` elements to persist
      between evaluations, or it may reset ``output_storage`` cells to
      hold a value of ``None``.  It can also pre-allocate some memory
      for the :class:`Op` to use.  This feature can allow :meth:`Op.perform` to reuse
      memory between calls, for example. If there is something
      preallocated in the ``output_storage``, it will be of the good
      dtype, but can have the wrong shape and have any stride pattern.

  This method must be determined by the inputs. That is to say, if
  it is evaluated once on inputs A and returned B, then if ever
  inputs C, equal to A, are presented again, then outputs equal to
  B must be returned again.

  You must be careful about aliasing outputs to inputs, and making
  modifications to any of the inputs. See :ref:`Views and inplace
  operations <views_and_inplace>` before writing a :meth:`Op.perform`
  implementation that does either of these things.


.. function:: __eq__(other)

  ``other`` is also an :class:`Op`.

  Returning ``True`` here is a promise to the rewrite system
  that the other :class:`Op` will produce exactly the same graph effects
  (e.g. from its :meth:`Op.perform`) as this one, given identical inputs. This means it
  will produce the same output values, it will destroy the same
  inputs (same :attr:`Op.destroy_map`), and will alias outputs to the same
  inputs (same :attr:`Op.view_map`). For more details, see
  :ref:`views_and_inplace`.

  .. note::

      If you set ``__props__``, this will be automatically generated.


.. function:: __hash__()

  If two :class:`Op` instances compare equal, then they **must** return the
  same hash value.

  Equally important, this hash value must not change during the
  lifetime of self.  :class:`Op` instances should be immutable in this
  sense.

.. note::

    If you set :attr:`Op.__props__`, this will be automatically generated.

.. op_optional:

Optional methods or attributes
==============================

.. attribute:: __props__

  Default: Undefined

  Must be a tuple.  Lists the name of the attributes which influence
  the computation performed.  This will also enable the automatic
  generation of appropriate ``__eq__``, ``__hash__`` and ``__str__`` methods.
  Should be set to ``()`` if you have no attributes that are relevant to
  the computation to generate the methods.

  .. versionadded:: 0.7

.. attribute:: default_output

  Default: None

  If this member variable is an integer, then the default
  implementation of ``__call__`` will return
  ``node.outputs[self.default_output]``, where ``node`` was returned
  by :meth:`Op.make_node`.  Otherwise, the entire list of outputs will be
  returned, unless it is of length 1, where the single element will be
  returned by itself.

.. function:: make_thunk(node, storage_map, compute_map, no_recycling, impl=None)

   This function must return a thunk, that is a zero-arguments
   function that encapsulates the computation to be performed by this
   :class:`Op` on the arguments of the node.

   :param node: :class:`Apply` instance
     The node for which a thunk is requested.
   :param storage_map: dict of lists
     This maps variables to a one-element lists holding the variable's
     current value. The one-element list acts as pointer to the value
     and allows sharing that "pointer" with other nodes and instances.
   :param compute_map: dict of lists
     This maps variables to one-element lists holding booleans.  If
     the value is 0 then the variable has not been computed and the
     value should not be considered valid.  If the value is 1 the
     variable has been computed and the value is valid.  If the value
     is 2 the variable has been garbage-collected and is no longer
     valid, but shouldn't be required anymore for this call.
   :param no_recycling: WRITEME
     WRITEME
   :param impl: None, 'c' or 'py'
     Which implementation to use.

   The returned function must ensure that is sets the computed
   variables as computed in the `compute_map`.

   Defining this function removes the requirement for :meth:`perform`
   or C code, as you will define the thunk for the computation
   yourself.

.. function:: __call__(*inputs, **kwargs)

   By default this is a convenience function which calls
   :meth:`make_node` with the supplied arguments and returns the
   result indexed by `default_output`.  This can be overridden by
   subclasses to do anything else, but must return either an Aesara
   :class:`Variable` or a list of :class:`Variable`\s.

   If you feel the need to override `__call__` to change the graph
   based on the arguments, you should instead create a function that
   will use your :class:`Op` and build the graphs that you want and call that
   instead of the :class:`Op` instance directly.

.. function:: infer_shape(fgraph, node, shapes)

   This function is needed for shape rewrites. ``shapes`` is a
   list with one tuple for each input of the :class:`Apply` node (which corresponds
   to the inputs of the :class:`Op`).  Each tuple contains as many elements as the
   number of dimensions of the corresponding input. The value of each element
   is the shape (number of items) along the corresponding dimension of that
   specific input.

   While this might sound complicated, it is nothing more than the shape
   of each input as symbolic variables (one per dimension).

   The function should return a list with one tuple for each output.
   Each tuple should contain the corresponding output's computed shape.

   Implementing this method will allow Aesara to compute the output's
   shape without computing the output itself, potentially sparing you
   a costly recomputation.

.. function:: flops(inputs, outputs)

   It is only used to have more information printed by the memory
   profiler.  It makes it print the mega flops and giga flops per
   second for each apply node. It takes as inputs two lists: one for the
   inputs and one for the outputs. They contain tuples that are the
   shapes of the corresponding inputs/outputs.

.. function:: __str__()

   This allows you to specify a more informative string representation of your
   :class:`Op`. If an `Op` has parameters, it is highly recommended to have the
   ``__str__`` method include the name of the :class:`Op` and the :Class:`Op`'s parameters'
   values.

   .. note::

     If you set `__props__`, this will be automatically generated.
     You can still override it for custom output.

.. function:: do_constant_folding(fgraph, node)

   Default: Return ``True``

   By default when rewrites are enabled, we remove during
   function compilation :class:`Apply` nodes whose inputs are all constants.
   We replace the :class:`Apply` node with an Aesara constant variable.
   This way, the :class:`Apply` node is not executed at each function
   call. If you want to force the execution of an :class:`Op` during the
   function call, make do_constant_folding return False.

   As done in the Alloc :class:`Op`, you can return False only in some cases by
   analyzing the graph from the node parameter.

.. function:: debug_perform(node, inputs, output_storage)

   Undefined by default.

   If you define this function then it will be used instead of C code
   or :meth:`Op.perform` to do the computation while debugging (currently
   DebugMode, but others may also use it in the future).  It has the
   same signature and contract as :meth:`Op.perform`.

   This enables :class:`Op`\s that cause trouble with DebugMode with their
   normal behaviour to adopt a different one when run under that
   mode. If your :class:`Op` doesn't have any problems, don't implement this.

If you want your :class:`Op` to work with :func:`aesara.gradient.grad` you also
need to implement the functions described below.

Gradient
========

These are the function required to work with :func:`aesara.gradient.grad`.

.. function:: grad(inputs, output_gradients)

  If the :class:`Op` being defined is differentiable, its gradient may be
  specified symbolically in this method. Both ``inputs`` and
  ``output_gradients`` are lists of symbolic Aesara :class:`Variable`\s and
  those must be operated on using Aesara's symbolic language. The :meth:`Op.grad`
  method must return a list containing one :class:`Variable` for each
  input. Each returned :class:`Variable` represents the gradient with respect
  to that input computed based on the symbolic gradients with respect
  to each output.

  If the output is not differentiable with respect to an input then
  this method should be defined to return a variable of type :class:`NullType`
  for that input. Likewise, if you have not implemented the gradient
  computation for some input, you may return a variable of type
  :class:`NullType` for that input. :mod:`aesara.gradient` contains convenience
  methods that can construct the variable for you:
  :func:`aesara.gradient.grad_undefined` and
  :func:`aesara.gradient.grad_not_implemented`, respectively.

  If an element of ``output_gradient`` is of type
  :class:`aesara.gradient.DisconnectedType`, it means that the cost is not a
  function of this output. If any of the :class:`Op`'s inputs participate in
  the computation of only disconnected outputs, then :meth:`Op.grad` should
  return :class:`DisconnectedType` variables for those inputs.

  If the :meth:`Op.grad` method is not defined, then Aesara assumes it has been
  forgotten.  Symbolic differentiation will fail on a graph that
  includes this :class:`Op`.

  It must be understood that the :meth:`Op.grad` method is not meant to
  return the gradient of the :class:`Op`'s output. :func:`aesara.grad` computes
  gradients; :meth:`Op.grad` is a helper function that computes terms that
  appear in gradients.

  If an :class:`Op` has a single vector-valued output ``y`` and a single
  vector-valued input ``x``, then the :meth:`Op.grad` method will be passed ``x`` and a
  second vector ``z``. Define ``J`` to be the Jacobian of ``y`` with respect to
  ``x``. The :meth:`Op.grad` method should return ``dot(J.T,z)``. When
  :func:`aesara.grad` calls the :meth:`Op.grad` method, it will set ``z`` to be the
  gradient of the cost ``C`` with respect to ``y``. If this :class:`Op` is the only :class:`Op`
  that acts on ``x``, then ``dot(J.T,z)`` is the gradient of C with respect to
  ``x``.  If there are other :class:`Op`\s that act on ``x``, :func:`aesara.grad` will
  have to add up the terms of ``x``'s gradient contributed by the other
  :meth:`Op.grad` method.

  In practice, an :class:`Op`'s input and output are rarely implemented as
  single vectors.  Even if an :class:`Op`'s output consists of a list
  containing a scalar, a sparse matrix, and a 4D tensor, you can think
  of these objects as being formed by rearranging a vector. Likewise
  for the input. In this view, the values computed by the :meth:`Op.grad` method
  still represent a Jacobian-vector product.

  In practice, it is probably not a good idea to explicitly construct
  the Jacobian, which might be very large and very sparse. However,
  the returned value should be equal to the Jacobian-vector product.

  So long as you implement this product correctly, you need not
  understand what :func:`aesara.gradient.grad` is doing, but for the curious the
  mathematical justification is as follows:

  In essence, the :meth:`Op.grad` method must simply implement through symbolic
  :class:`Variable`\s and operations the chain rule of differential
  calculus. The chain rule is the mathematical procedure that allows
  one to calculate the total derivative :math:`\frac{d C}{d x}` of the
  final scalar symbolic `Variable` ``C`` with respect to a primitive
  symbolic :class:`Variable` x found in the list ``inputs``.  The :meth:`Op.grad` method
  does this using ``output_gradients`` which provides the total
  derivative :math:`\frac{d C}{d f}` of ``C`` with respect to a symbolic
  :class:`Variable` that is returned by the `Op` (this is provided in
  ``output_gradients``), as well as the knowledge of the total
  derivative :math:`\frac{d f}{d x}` of the latter with respect to the
  primitive :class:`Variable` (this has to be computed).

  In mathematics, the total derivative of a scalar variable :math:`C` with
  respect to a vector of scalar variables :math:`x`, i.e. the gradient, is
  customarily represented as the row vector of the partial
  derivatives, whereas the total derivative of a vector of scalar
  variables :math:`f` with respect to another :math:`x`, is customarily
  represented by the matrix of the partial derivatives, i.e. the
  Jacobian matrix. In this convenient setting, the chain rule
  says that the gradient of the final scalar variable :math:`C` with
  respect to the primitive scalar variables in :math:`x` through those in
  :math:`f` is simply given by the matrix product:
  :math:`\frac{d C}{d x} = \frac{d C}{d f} * \frac{d f}{d x}`.

  Here, the chain rule must be implemented in a similar but slightly
  more complex setting: Aesara provides in the list
  ``output_gradients`` one gradient for each of the :class:`Variable`\s returned
  by the `Op`. Where :math:`f` is one such particular :class:`Variable`, the
  corresponding gradient found in ``output_gradients`` and
  representing :math:`\frac{d C}{d f}` is provided with a shape
  similar to :math:`f` and thus not necessarily as a row vector of scalars.
  Furthermore, for each :class:`Variable` :math:`x` of the :class:`Op`'s list of input variables
  ``inputs``, the returned gradient representing :math:`\frac{d C}{d
  x}` must have a shape similar to that of :class:`Variable` x.

  If the output list of the :class:`Op` is :math:`[f_1, ... f_n]`, then the
  list ``output_gradients`` is :math:`[grad_{f_1}(C), grad_{f_2}(C),
  ... , grad_{f_n}(C)]`.  If ``inputs`` consists of the list
  :math:`[x_1, ..., x_m]`, then `Op.grad` should return the list
  :math:`[grad_{x_1}(C), grad_{x_2}(C), ..., grad_{x_m}(C)]`, where
  :math:`(grad_{y}(Z))_i = \frac{\partial Z}{\partial y_i}` (and
  :math:`i` can stand for multiple dimensions).

  In other words, :meth:`Op.grad` does not return :math:`\frac{d f_i}{d
  x_j}`, but instead the appropriate dot product specified by the
  chain rule: :math:`\frac{d C}{d x_j} = \frac{d C}{d f_i} \cdot
  \frac{d f_i}{d x_j}`.  Both the partial differentiation and the
  multiplication have to be performed by :meth:`Op.grad`.

  Aesara currently imposes the following constraints on the values
  returned by the :meth:`Op.grad` method:

  1) They must be :class:`Variable` instances.
  2) When they are types that have dtypes, they must never have an integer dtype.

  The output gradients passed *to* :meth:`Op.grad` will also obey these constraints.

  Integers are a tricky subject. Integers are the main reason for
  having :class:`DisconnectedType`, :class:`NullType` or zero gradient. When you have an
  integer as an argument to your :meth:`Op.grad` method, recall the definition of
  a derivative to help you decide what value to return:

  :math:`\frac{d f}{d x} = \lim_{\epsilon \rightarrow 0} (f(x+\epsilon)-f(x))/\epsilon`.

  Suppose your function f has an integer-valued output. For most
  functions you're likely to implement in Aesara, this means your
  gradient should be zero, because :math:`f(x+epsilon) = f(x)` for almost all
  :math:`x`. (The only other option is that the gradient could be undefined,
  if your function is discontinuous everywhere, like the rational
  indicator function)

  Suppose your function :math:`f` has an integer-valued input. This is a
  little trickier, because you need to think about what you mean
  mathematically when you make a variable integer-valued in
  Aesara. Most of the time in machine learning we mean ":math:`f` is a
  function of a real-valued :math:`x`, but we are only going to pass in
  integer-values of :math:`x`". In this case, :math:`f(x+\epsilon)` exists, so the
  gradient through :math:`f` should be the same whether :math:`x` is an integer or a
  floating point variable. Sometimes what we mean is ":math:`f` is a function
  of an integer-valued :math:`x`, and :math:`f` is only defined where :math:`x` is an
  integer." Since :math:`f(x+\epsilon)` doesn't exist, the gradient is
  undefined.  Finally, many times in Aesara, integer valued inputs
  don't actually affect the elements of the output, only its shape.

  If your function :math:`f` has both an integer-valued input and an
  integer-valued output, then both rules have to be combined:

  - If :math:`f` is defined at :math:`x + \epsilon`, then the input gradient is
    defined. Since :math:`f(x+\epsilon)` would be equal to :math:`f(x)` almost
    everywhere, the gradient should be zero (first rule).

  - If :math:`f` is only defined where :math:`x` is an integer, then the gradient
    is undefined, regardless of what the gradient with respect to the
    output is.

  Examples:

  1) :math:`f(x,y)` is a dot product between :math:`x` and :math:`y`. :math:`x` and :math:`y` are integers.
     Since the output is also an integer, :math:`f` is a step function.
     Its gradient is zero almost everywhere, so :meth:`Op.grad` should return
     zeros in the shape of :math:`x` and :math:`y`.
  2) :math:`f(x,y)` is a dot product between :math:`x` and :math:`y`. :math:`x`
     is floating point and :math:`y` is an integer.  In this case the output is
     floating point. It doesn't matter that :math:`y` is an integer.  We
     consider :math:`f` to still be defined at :math:`f(x,y+\epsilon)`. The
     gradient is exactly the same as if :math:`y` were floating point.
  3) :math:`f(x,y)` is the argmax of :math:`x` along axis :math:`y`.  The
     gradient with respect to :math:`y` is undefined, because :math:`f(x,y)` is
     not defined for floating point :math:`y`. How could you take an argmax
     along a fractional axis?  The gradient with respect to :math:`x` is 0,
     because :math:`f(x+\epsilon, y) = f(x)` almost everywhere.
  4) :math:`f(x,y)` is a vector with :math:`y` elements, each of which taking on
     the value :math:`x` The :meth:`Op.grad` method should return
     :class:`DisconnectedType` for :math:`y`, because the elements of :math:`f`
     don't depend on :math:`y`. Only the shape of :math:`f` depends on
     :math:`y`. You probably also want to implement a connection_pattern method to encode this.
  5) :math:`f(x) = int(x)` converts float :math:`x` into an integer. :math:`g(y) = float(y)`
     converts an integer :math:`y` into a float.  If the final cost :math:`C = 0.5 *
     g(y) = 0.5 g(f(x))`, then the gradient with respect to :math:`y` will be 0.5,
     even if :math:`y` is an integer. However, the gradient with respect to :math:`x` will be
     0, because the output of :math:`f` is integer-valued.

.. function:: connection_pattern(node):

  Sometimes needed for proper operation of :func:`aesara.gradient.grad`.

  Returns a list of list of booleans.

  ``Op.connection_pattern[input_idx][output_idx]`` is true if the
  elements of ``inputs[input_idx]`` have an effect on the elements of
  ``outputs[output_idx]``.

  The ``node`` parameter is needed to determine the number of inputs. Some
  :class:`Op`\s such as :class:`Subtensor` take a variable number of inputs.

  If no connection_pattern is specified, :func:`aesara.gradient.grad` will
  assume that all inputs have some elements connected to some
  elements of all outputs.

  This method conveys two pieces of information that are otherwise
  not part of the Aesara graph:

  1) Which of the :class:`Op`'s inputs are truly ancestors of each of the
     :class:`Op`'s outputs. Suppose an :class:`Op` has two inputs, :math:`x` and :math:`y`, and
     outputs :math:`f(x)` and :math:`g(y)`. :math:`y` is not really an ancestor of :math:`f`, but
     it appears to be so in the Aesara graph.
  2) Whether the actual elements of each input/output are relevant to a
     computation.
     For example, the shape :class:`Op` does not read its input's elements,
     only its shape metadata. :math:`\frac{d shape(x)}{dx}` should thus raise
     a disconnected input exception (if these exceptions are enabled).
     As another example, the elements of the :class:`Alloc` :class:`Op`'s outputs
     are not affected by the shape arguments to the :class:`Alloc` :class:`Op`.

  Failing to implement this function for an :class:`Op` that needs it can
  result in two types of incorrect behavior:

  1) :func:`aesara.gradient.grad` erroneously raising a ``TypeError`` reporting that
     a gradient is undefined.
  2) :func:`aesara.gradient.grad` failing to raise a ``ValueError`` reporting that
     an input is disconnected.

  Even if connection_pattern is not implemented correctly, if
  :func:`aesara.gradient.grad` returns an expression, that expression will be
  numerically correct.

.. function:: R_op(inputs, eval_points)

   Optional, to work with :func:`aesara.gradient.R_op`.

   This function implements the application of the R-operator on the
   function represented by your :class:`Op`. Let assume that function is :math:`f`,
   with input :math:`x`, applying the R-operator means computing the
   Jacobian of :math:`f` and right-multiplying it by :math:`v`, the evaluation
   point, namely: :math:`\frac{\partial f}{\partial x} v`.

   ``inputs`` are the symbolic variables corresponding to the value of
   the input where you want to evaluate the Jacobian, and ``eval_points``
   are the symbolic variables corresponding to the value you want to
   right multiply the Jacobian with.

   Same conventions as for the :meth:`Op.grad` method hold. If your :class:`Op`
   is not differentiable, you can return None. Note that in contrast to the
   method :meth:`Op.grad`, for :meth:`Op.R_op` you need to return the
   same number of outputs as there are outputs of the :class:`Op`. You can think
   of it in the following terms. You have all your inputs concatenated
   into a single vector :math:`x`. You do the same with the evaluation
   points (which are as many as inputs and of the shame shape) and obtain
   another vector :math:`v`. For each output, you reshape it into a vector,
   compute the Jacobian of that vector with respect to :math:`x` and
   multiply it by :math:`v`. As a last step you reshape each of these
   vectors you obtained for each outputs (that have the same shape as
   the outputs) back to their corresponding shapes and return them as the
   output of the :meth:`Op.R_op` method.

   :ref:`List of op with r op support <R_op_list>`.

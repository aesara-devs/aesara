from collections.abc import Sequence
from copy import copy
from warnings import warn

import numpy as np

import theano
from theano.compile.ops import Shape
from theano.gof.graph import Apply, Variable
from theano.gof.op import Op
from theano.tensor.basic import (
    NotScalarConstantError,
    as_tensor_variable,
    constant,
    get_scalar_constant_value,
    get_vector_length,
    int_dtypes,
)
from theano.tensor.raw_random import RandomStateType
from theano.tensor.subtensor import Subtensor
from theano.tensor.type import TensorType
from theano.tensor.type_other import NoneConst


def param_supp_shape_fn(
    ndim_supp, ndims_params, dist_params, rep_param_idx=0, param_shapes=None
):
    """Infer dimensions for a random variable.

    This is a function that derives a random variable's support
    shape/dimensions from one of its parameters.

    XXX: It's not always possible to determine a random variable's support
    shape from its parameters, so this function has fundamentally limited
    applicability.

    XXX: This function is not expected to handle `ndim_supp = 0` (i.e.
    scalars), since that is already definitively handled in the `Op` that
    calls this.

    TODO: Consider using `theano.compile.ops.shape_i` alongside `ShapeFeature`.

    Parameters
    ----------
    ndim_supp: int
        Total number of dimensions in the support (assumedly > 0).
    ndims_params: list of int
        Number of dimensions for each distribution parameter.
    dist_params: list of `theano.gof.graph.Variable`
        The distribution parameters.
    param_shapes: list of `theano.compile.ops.Shape` (optional)
        Symbolic shapes for each distribution parameter.
        Providing this value prevents us from reproducing the requisite
        `theano.compile.ops.Shape` object (e.g. when it's already available to
        the caller).
    rep_param_idx: int (optional)
        The index of the distribution parameter to use as a reference
        In other words, a parameter in `dist_param` with a shape corresponding
        to the support's shape.
        The default is the first parameter (i.e. the value 0).

    Results
    -------
    out: a tuple representing the support shape for a distribution with the
    given `dist_params`.

    """
    # XXX: Gotta be careful slicing Theano variables, the `Subtensor` Op isn't
    # handled by `tensor.get_scalar_constant_value`!
    # E.g.
    #     test_val = as_tensor_variable([[1], [4]])
    #     get_scalar_constant_value(test_val.shape[-1]) # works
    #     get_scalar_constant_value(test_val.shape[0]) # doesn't
    #     get_scalar_constant_value(test_val.shape[:-1]) # doesn't
    if param_shapes is not None:
        ref_param = param_shapes[rep_param_idx]
        return (ref_param[-ndim_supp],)
    else:
        ref_param = dist_params[rep_param_idx]
        if ref_param.ndim < ndim_supp:
            raise ValueError(
                (
                    "Reference parameter does not match the "
                    "expected dimensions; {} has less than {} dim(s).".format(
                        ref_param, ndim_supp
                    )
                )
            )
        return (ref_param.shape[-ndim_supp],)


class RandomVariable(Op):
    """An `Op` that produces a sample from a random variable.

    This is essentially `RandomFunction`, except that it removes the
    `outtype` dependency and handles shape dimension information more
    directly.

    """

    __props__ = ("name", "dtype", "ndim_supp", "inplace", "ndims_params")
    default_output = 1
    nondeterministic = True

    def __init__(
        self,
        name,
        ndim_supp,
        ndims_params,
        rng_fn,
        *args,
        supp_shape_fn=param_supp_shape_fn,
        dtype=None,
        inplace=False,
        **kwargs,
    ):
        """Create a random variable `Op`.

        Parameters
        ----------
        name: str
            The `Op`'s display name.
        ndim_supp: int
            Dimension of the support.  This value is used to infer the exact
            shape of the support and independent terms from ``dist_params``.
        ndims_params: tuple (int)
            Number of dimensions of each parameter in ``dist_params``.
        rng_fn: function or str
            The non-symbolic random variate sampling function.
            Can be the string name of a method provided by
            `numpy.random.RandomState`.
        supp_shape_fn: callable (optional)
            Function used to determine the exact shape of the distribution's
            support.

            It must take arguments ndim_supp, ndims_params, dist_params
            (i.e. an collection of the distribution parameters) and an
            optional param_shapes (i.e. tuples containing the size of each
            dimension for each distribution parameter).

            Defaults to `param_supp_shape_fn`.
        dtype: Theano dtype (optional)
            The dtype of the sampled output(s).  If `None` (the default), the
            `dtype` keyword must be set when `RandomVariable.make_node` is
            called.
        inplace: boolean (optional)
            Determine whether or not the underlying rng state is updated
            in-place or not (i.e. copied).

        """
        super().__init__(*args, **kwargs)

        self.name = name
        self.ndim_supp = ndim_supp
        self.supp_shape_fn = supp_shape_fn
        self.dtype = dtype
        self.inplace = inplace

        if not isinstance(ndims_params, Sequence):
            raise TypeError("Parameter ndims_params must be sequence type.")

        self.ndims_params = tuple(ndims_params)

        if isinstance(rng_fn, str):
            self.rng_fn = getattr(np.random.RandomState, rng_fn)
        else:
            self.rng_fn = rng_fn

    def __str__(self):
        return "{}_rv".format(self.name)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        """Compute the output shape given the size and distribution parameters.

        Outputs
        -------
        shape : tuple of ScalarVariable

        """

        param_shapes = param_shapes or [p.shape for p in dist_params]

        def slice_ind_dims(p, ps, n):
            shape = tuple(ps)

            if n == 0:
                return (p, shape, p.broadcastable)

            ind_slice = (np.s_[:],) * (p.ndim - n) + (0,) * n
            return (p[ind_slice], shape[:-n], p.broadcastable[:-n])

        # These are versions of our actual parameters with the anticipated
        # dimensions (i.e. support dimensions) removed so that only the
        # independent variate dimensions are left.
        params_ind_slice = tuple(
            slice_ind_dims(p, ps, n)
            for p, ps, n in zip(dist_params, param_shapes, self.ndims_params)
        )

        if len(params_ind_slice) == 1:
            ind_param, ind_shape, ind_bcast = params_ind_slice[0]
            ndim_ind = len(ind_shape)
            shape_ind = ind_shape
        elif len(params_ind_slice) > 1:
            # If there are multiple parameters, the dimensions of their
            # independent variates should broadcast together.
            p_slices, p_shapes, p_bcasts = zip(*params_ind_slice)
            shape_ind = theano.tensor.extra_ops.broadcast_shape_iter(
                p_shapes, arrays_are_shapes=True
            )
            ndim_ind = len(shape_ind)

        size_len = get_vector_length(size)

        if self.ndim_supp == 0:
            shape_supp = tuple()

            # In the scalar case, `size` corresponds to the entire result's
            # shape. This implies the following:
            #     shape_ind == size[:ndim_ind]
            # TODO: Do we want to constraint/check symbolically?  We could use
            # an `Assert`, but we could also let the numeric sampler handle the
            # error.

            shape_reps = tuple(size)

            if ndim_ind > 0:
                shape_reps = shape_reps[:-ndim_ind]

            ndim_reps = len(shape_reps)
        else:
            shape_supp = self.supp_shape_fn(
                self.ndim_supp,
                self.ndims_params,
                dist_params,
                param_shapes=param_shapes,
            )

            ndim_reps = size_len
            shape_reps = size

        ndim_shape = self.ndim_supp + ndim_ind + ndim_reps

        if ndim_shape == 0:
            shape = constant([], dtype="int64")
        else:
            shape = tuple(shape_reps) + tuple(shape_ind) + tuple(shape_supp)

        # if shape is None:
        #     raise ShapeError()

        return shape

    @theano.change_flags(compute_test_value="off")
    def compute_bcast(self, dist_params, size):
        """Compute the broadcast array for this distribution's `TensorType`.

        Parameters
        ----------
        dist_params: list
            Distribution parameters.
        size: int or Sequence (optional)
            Numpy-like size of the output (i.e. replications).

        """
        shape = self._infer_shape(size, dist_params)

        # Let's try to do a better job than `_infer_ndim_bcast` when
        # dimension sizes are symbolic.
        bcast = []
        for s in shape:
            s_owner = getattr(s, "owner", None)

            # Get rid of the `Assert`s added by `broadcast_shape`
            if s_owner and isinstance(s_owner.op, theano.tensor.opt.Assert):
                s_owner = s_owner.inputs[0].owner

            try:
                if (
                    s_owner
                    and isinstance(s_owner.op, Subtensor)
                    and s_owner.inputs[0].owner is not None
                ):
                    # Handle a special case in which
                    # `tensor.get_scalar_constant_value` doesn't really work.
                    s_x, s_idx = s_owner.inputs
                    s_idx = get_scalar_constant_value(s_idx)
                    if isinstance(s_x.owner.op, Shape):
                        (x_obj,) = s_x.owner.inputs
                        s_val = x_obj.type.broadcastable[s_idx]
                    else:
                        # TODO: Could go for an existing broadcastable here,
                        # too, no?
                        s_val = False
                else:
                    s_val = get_scalar_constant_value(s)
            except NotScalarConstantError:
                s_val = False

            bcast += [s_val == 1]
        return bcast

    def infer_shape(self, node, input_shapes):
        size = node.inputs[-2]
        dist_params = tuple(node.inputs[:-2])
        shape = self._infer_shape(size, dist_params, param_shapes=input_shapes[:-2])

        return [None, [s for s in shape]]

    def make_node(self, *dist_params, size=None, rng=None, name=None, dtype=None):
        """Create a random variable node.

        XXX: Unnamed/non-keyword arguments are considered distribution
        parameters!  If you want to set `size`, `rng`, and/or `name`, use their
        keywords.

        Parameters
        ----------
        dist_params: list
            Distribution parameters.
        size: int or Sequence (optional)
            Numpy-like size of the output (i.e. replications).
        rng: RandomState (optional)
            Existing Theano `RandomState` object to be used.  Creates a
            new one, if `None`.
        name: str (optional)
            Label for the resulting node.
        dtype: Theano dtype (optional)
            The dtype of the sampled output.  This value is only used when
            `self.dtype` isn't set.

        Results
        -------
        out: `Apply`
            A node with inputs `dist_args + (size, in_rng, name)` and outputs
            `(out_rng, sample_tensorvar)`.

        """
        if size is None:
            size = constant([], dtype="int64")
        elif isinstance(size, int):
            size = as_tensor_variable([size], ndim=1)
        elif not isinstance(size, (np.ndarray, Variable, Sequence)):
            raise TypeError(
                "Parameter size must be None, an integer, or a sequence with integers."
            )
        else:
            size = as_tensor_variable(size, ndim=1)

        assert size.dtype in int_dtypes

        dist_params = tuple(
            as_tensor_variable(p) if not isinstance(p, Variable) else p
            for p in dist_params
        )

        if rng is None:
            rng = theano.shared(np.random.RandomState())
        elif not isinstance(rng.type, RandomStateType):
            warn("The type of rng should be an instance of RandomStateType")

        bcast = self.compute_bcast(dist_params, size)
        dtype = self.dtype or dtype

        if dtype is None:
            # dtype = tt.scal.upcast(self.dtype, *[p.dtype for p in dist_params])
            raise TypeError("dtype is unspecified")

        outtype = TensorType(dtype=dtype, broadcastable=bcast)
        out_var = outtype(name=name)
        inputs = dist_params + (size, rng)
        outputs = (rng.type(), out_var)

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Draw samples using Numpy/SciPy."""
        rng_out, smpl_out = outputs

        args = list(inputs)
        rng = args.pop()
        size = args.pop()

        assert isinstance(rng, np.random.RandomState), (type(rng), rng)

        rng_out[0] = rng

        # The symbolic output variable corresponding to value produced here.
        out_var = node.outputs[1]

        # If `size == []`, that means no size is enforced, and NumPy is
        # trusted to draw the appropriate number of samples, NumPy uses
        # `size=None` to represent that.  Otherwise, NumPy expects a tuple.
        if np.size(size) == 0:
            size = None
        else:
            size = tuple(size)

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng`
        # otherwise.
        if not self.inplace:
            rng = copy(rng)

        smpl_val = self.rng_fn(rng, *(args + [size]))

        if (
            not isinstance(smpl_val, np.ndarray)
            or str(smpl_val.dtype) != out_var.type.dtype
        ):
            smpl_val = theano._asarray(smpl_val, dtype=out_var.type.dtype)

        # When `size` is `None`, NumPy has a tendency to unexpectedly
        # return a scalar instead of a higher-dimension array containing
        # only one element. This value should be reshaped
        # TODO: Really?  Why shouldn't the output correctly correspond to
        # the returned NumPy value?  Sounds more like a mis-specification of
        # the symbolic output variable.
        if size is None and smpl_val.ndim == 0 and out_var.ndim > 0:
            smpl_val = smpl_val.reshape([1] * out_var.ndim)

        smpl_out[0] = smpl_val

    def grad(self, inputs, outputs):
        return [
            theano.gradient.grad_undefined(
                self, k, inp, "No gradient defined through raw random numbers op"
            )
            for k, inp in enumerate(inputs)
        ]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]


class Observed(Op):
    """An `Op` that represents an observed random variable.

    This `Op` establishes an observation relationship between a random
    variable and a specific value.
    """

    default_output = 0

    def __init__(self):
        self.view_map = {0: [0]}

    def make_node(self, val, rv=None):
        """Make an `Observed` random variable.

        Parameters
        ----------
        val: Variable
            The observed value.
        rv: RandomVariable
            The distribution from which `val` is assumed to be a sample value.
        """
        val = as_tensor_variable(val)
        if rv is not None:
            if not hasattr(rv, "type") or rv.type.convert_variable(val) is None:
                raise ValueError(
                    (
                        "`rv` and `val` do not have compatible types:"
                        f" rv={rv}, val={val}"
                    )
                )
        else:
            rv = NoneConst.clone()

        inputs = [val, rv]

        return Apply(self, inputs, [val.type()])

    def perform(self, node, inputs, out):
        out[0][0] = inputs[0]

    def grad(self, inputs, outputs):
        return outputs


observed = Observed()

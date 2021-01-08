from collections.abc import Sequence
from copy import copy

import numpy as np

import theano
from theano.configdefaults import config
from theano.graph.basic import Apply, Variable
from theano.graph.op import Op
from theano.misc.safe_asarray import _asarray
from theano.tensor.basic import (
    NotScalarConstantError,
    all_dtypes,
    as_tensor_variable,
    cast,
    constant,
    get_scalar_constant_value,
    get_vector_length,
    int_dtypes,
)
from theano.tensor.random.type import RandomStateType
from theano.tensor.random.utils import params_broadcast_shapes
from theano.tensor.type import TensorType
from theano.tensor.type_other import NoneConst


def default_shape_from_params(
    ndim_supp, dist_params, rep_param_idx=0, param_shapes=None
):
    """Infer the dimensions for the output of a `RandomVariable`.

    This is a function that derives a random variable's support
    shape/dimensions from one of its parameters.

    XXX: It's not always possible to determine a random variable's support
    shape from its parameters, so this function has fundamentally limited
    applicability and must be replaced by custom logic in such cases.

    XXX: This function is not expected to handle `ndim_supp = 0` (i.e.
    scalars), since that is already definitively handled in the `Op` that
    calls this.

    TODO: Consider using `theano.compile.ops.shape_i` alongside `ShapeFeature`.

    Parameters
    ----------
    ndim_supp: int
        Total number of dimensions for a single draw of the random variable
        (e.g. a multivariate normal draw is 1D, so `ndim_supp = 1`).
    dist_params: list of `theano.graph.basic.Variable`
        The distribution parameters.
    param_shapes: list of tuple of `ScalarVariable` (optional)
        Symbolic shapes for each distribution parameter.  These will
        be used in place of distribution parameter-generated shapes.
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
    if ndim_supp <= 0:
        raise ValueError("ndim_supp must be greater than 0")
    if param_shapes is not None:
        ref_param = param_shapes[rep_param_idx]
        return (ref_param[-ndim_supp],)
    else:
        ref_param = dist_params[rep_param_idx]
        if ref_param.ndim < ndim_supp:
            raise ValueError(
                (
                    "Reference parameter does not match the "
                    f"expected dimensions; {ref_param} has less than {ndim_supp} dim(s)."
                )
            )
        return ref_param.shape[-ndim_supp:]


class RandomVariable(Op):
    """An `Op` that produces a sample from a random variable.

    This is essentially `RandomFunction`, except that it removes the
    `outtype` dependency and handles shape dimension information more
    directly.

    """

    __props__ = ("name", "ndim_supp", "ndims_params", "dtype", "inplace")
    default_output = 1
    nondeterministic = True

    def __init__(
        self,
        name=None,
        ndim_supp=None,
        ndims_params=None,
        dtype=None,
        inplace=None,
    ):
        """Create a random variable `Op`.

        Parameters
        ----------
        name: str
            The `Op`'s display name.
        ndim_supp: int
            Total number of dimensions for a single draw of the random variable
            (e.g. a multivariate normal draw is 1D, so ``ndim_supp = 1``).
        ndims_params: list of int
            Number of dimensions for each distribution parameter when the
            parameters only specify a single drawn of the random variable
            (e.g. a multivariate normal's mean is 1D and covariance is 2D, so
            ``ndims_params = [1, 2]``).
        dtype: str (optional)
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then ``dtype`` is set to ``theano.config.floatX``.  If
            ``None`` (the default), the `dtype` keyword must be set when
            `RandomVariable.make_node` is called.
        inplace: boolean (optional)
            Determine whether or not the underlying rng state is updated
            in-place or not (i.e. copied).

        """
        super().__init__()

        self.name = name or getattr(self, "name")
        self.ndim_supp = (
            ndim_supp if ndim_supp is not None else getattr(self, "ndim_supp")
        )
        self.ndims_params = (
            ndims_params if ndims_params is not None else getattr(self, "ndims_params")
        )
        self.dtype = dtype or getattr(self, "dtype", None)

        self.inplace = (
            inplace if inplace is not None else getattr(self, "inplace", False)
        )

        if not isinstance(self.ndims_params, Sequence):
            raise TypeError("Parameter ndims_params must be sequence type.")

        self.ndims_params = tuple(self.ndims_params)

        if self.inplace:
            self.destroy_map = {0: [len(self.ndims_params) + 1]}

    def _shape_from_params(self, dist_params, **kwargs):
        """Determine the shape of a `RandomVariable`'s output given its parameters.

        This does *not* consider the extra dimensions added by the `size` parameter.

        Defaults to `param_supp_shape_fn`.
        """
        return default_shape_from_params(self.ndim_supp, dist_params, **kwargs)

    def rng_fn(self, rng, *args, **kwargs):
        """Sample a numeric random variate."""
        return getattr(np.random.RandomState, self.name)(rng, *args, **kwargs)

    def __str__(self):
        return "{}_rv".format(self.name)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        """Compute the output shape given the size and distribution parameters.

        Parameters
        ----------
        size : TensorVariable
            The size parameter specified for this `RandomVariable`.
        dist_params : list of TensorVariable
            The symbolic parameter for this `RandomVariable`'s distribution.
        param_shapes : list of tuples of TensorVariable (optional)
            The shapes of the `dist_params` as given by `ShapeFeature`'s
            via `Op.infer_shape`'s `input_shapes` argument.  This parameter's
            values are essentially more accurate versions of ``[d.shape for d
            in dist_params]``.

        Outputs
        -------
        shape : tuple of `ScalarVariable`

        """

        size_len = get_vector_length(size)

        if self.ndim_supp == 0 and size_len > 0:
            # In this case, we have a univariate distribution with a non-empty
            # `size` parameter, which means that the `size` parameter
            # completely determines the shape of the random variable.  More
            # importantly, the `size` parameter may be the only correct source
            # of information for the output shape, in that we would be misled
            # by the `dist_params` if we tried to infer the relevant parts of
            # the output shape from those.
            return size

        # Broadcast the parameters
        param_shapes = params_broadcast_shapes(
            param_shapes or [p.shape for p in dist_params], self.ndims_params
        )

        def slice_ind_dims(p, ps, n):
            shape = tuple(ps)

            if n == 0:
                return (p, shape)

            ind_slice = (slice(None),) * (p.ndim - n) + (0,) * n
            ind_shape = [
                s if b is False else constant(1, "int64")
                for s, b in zip(shape[:-n], p.broadcastable[:-n])
            ]
            return (
                p[ind_slice],
                ind_shape,
            )

        # These are versions of our actual parameters with the anticipated
        # dimensions (i.e. support dimensions) removed so that only the
        # independent variate dimensions are left.
        params_ind_slice = tuple(
            slice_ind_dims(p, ps, n)
            for p, ps, n in zip(dist_params, param_shapes, self.ndims_params)
        )

        if len(params_ind_slice) == 1:
            ind_param, ind_shape = params_ind_slice[0]
            ndim_ind = len(ind_shape)
            shape_ind = ind_shape
        elif len(params_ind_slice) > 1:
            # If there are multiple parameters, the dimensions of their
            # independent variates should broadcast together.
            p_slices, p_shapes = zip(*params_ind_slice)

            shape_ind = theano.tensor.extra_ops.broadcast_shape_iter(
                p_shapes, arrays_are_shapes=True
            )

            ndim_ind = len(shape_ind)
        else:
            ndim_ind = 0

        if self.ndim_supp == 0:
            shape_supp = tuple()
            shape_reps = tuple(size)

            if ndim_ind > 0:
                shape_reps = shape_reps[:-ndim_ind]

            ndim_reps = len(shape_reps)
        else:
            shape_supp = self._shape_from_params(
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

    @config.change_flags(compute_test_value="off")
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
                s = s_owner.inputs[0]

            try:
                s_val = get_scalar_constant_value(s)
            except NotScalarConstantError:
                s_val = False

            bcast += [s_val == 1]
        return bcast

    def infer_shape(self, fgraph, node, input_shapes):
        _, size, _, *dist_params = node.inputs
        _, _, _, *param_shapes = input_shapes

        shape = self._infer_shape(size, dist_params, param_shapes=param_shapes)

        return [None, [s for s in shape]]

    def __call__(self, *args, size=None, name=None, rng=None, dtype=None, **kwargs):
        res = super().__call__(rng, size, dtype, *args, **kwargs)

        if name is not None:
            res.name = name

        return res

    def make_node(self, rng, size, dtype, *dist_params):
        """Create a random variable node.

        XXX: Unnamed/non-keyword arguments are considered distribution
        parameters!  If you want to set `size`, `rng`, and/or `name`, use their
        keywords.

        Parameters
        ----------
        rng: RandomStateType
            Existing Theano `RandomState` object to be used.  Creates a
            new one, if `None`.
        size: int or Sequence
            Numpy-like size of the output (i.e. replications).
        dtype: str
            The dtype of the sampled output.  If the value ``"floatX"`` is
            given, then ``dtype`` is set to ``theano.config.floatX``.  This
            value is only used when `self.dtype` isn't set.
        dist_params: list
            Distribution parameters.

        Results
        -------
        out: `Apply`
            A node with inputs `(rng, size, dtype) + dist_args` and outputs
            `(rng_var, out_var)`.

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
            size = cast(as_tensor_variable(size, ndim=1), "int64")

        assert size.dtype in int_dtypes

        dist_params = tuple(
            as_tensor_variable(p) if not isinstance(p, Variable) else p
            for p in dist_params
        )

        if rng is None:
            rng = theano.shared(np.random.RandomState())
        elif not isinstance(rng.type, RandomStateType):
            raise TypeError("The type of rng should be an instance of RandomStateType")

        bcast = self.compute_bcast(dist_params, size)
        dtype = self.dtype or dtype

        if dtype == "floatX":
            dtype = config.floatX
        elif dtype is None or (isinstance(dtype, str) and dtype not in all_dtypes):
            # dtype = tt.scal.upcast(self.dtype, *[p.dtype for p in dist_params])
            raise TypeError("dtype is unspecified")

        if isinstance(dtype, str):
            dtype_idx = constant(all_dtypes.index(dtype), dtype="int64")
        else:
            dtype_idx = constant(dtype, dtype="int64")
            dtype = all_dtypes[dtype_idx.data]

        outtype = TensorType(dtype=dtype, broadcastable=bcast)
        out_var = outtype()
        inputs = (rng, size, dtype_idx) + dist_params
        outputs = (rng.type(), out_var)

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        rng_var_out, smpl_out = outputs

        rng, size, dtype, *args = inputs

        out_var = node.outputs[1]

        # If `size == []`, that means no size is enforced, and NumPy is trusted
        # to draw the appropriate number of samples, NumPy uses `size=None` to
        # represent that.  Otherwise, NumPy expects a tuple.
        if np.size(size) == 0:
            size = None
        else:
            size = tuple(size)

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng`
        # otherwise.
        if not self.inplace:
            rng = copy(rng)

        rng_var_out[0] = rng

        smpl_val = self.rng_fn(rng, *(args + [size]))

        if (
            not isinstance(smpl_val, np.ndarray)
            or str(smpl_val.dtype) != out_var.type.dtype
        ):
            smpl_val = _asarray(smpl_val, dtype=out_var.type.dtype)

        smpl_out[0] = smpl_val

    def grad(self, inputs, outputs):
        return [
            theano.gradient.grad_undefined(
                self, k, inp, "No gradient defined for random variables"
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
    view_map = {0: [1]}

    def make_node(self, rv, val):
        """Make an `Observed` random variable.

        Parameters
        ----------
        rv: RandomVariable
            The distribution from which `val` is assumed to be a sample value.
        val: Variable
            The observed value.
        """
        val = as_tensor_variable(val)

        if rv is not None:
            if not hasattr(rv, "type") or rv.type.convert_variable(val) is None:
                raise TypeError(
                    (
                        "`rv` and `val` do not have compatible types:"
                        f" rv={rv}, val={val}"
                    )
                )
        else:
            rv = NoneConst.clone()

        inputs = [rv, val]

        return Apply(self, inputs, [val.type()])

    def perform(self, node, inputs, out):
        out[0][0] = inputs[1]

    def grad(self, inputs, outputs):
        return [
            theano.gradient.grad_undefined(
                self, k, inp, "No gradient defined for random variables"
            )
            for k, inp in enumerate(inputs)
        ]


observed = Observed()

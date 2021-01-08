from functools import wraps
from itertools import zip_longest

import numpy as np

from theano.compile.sharedvalue import shared
from theano.graph.basic import Variable
from theano.tensor.basic import maximum
from theano.tensor.extra_ops import broadcast_to


def params_broadcast_shapes(param_shapes, ndims_params, use_theano=True):
    """Broadcast parameters that have different dimensions.

    Parameters
    ==========
    param_shapes : list of ndarray or Variable
        The shapes of each parameters to broadcast.
    ndims_params : list of int
        The expected number of dimensions for each element in `params`.
    use_theano : bool
        If ``True``, use Theano `Op`; otherwise, use NumPy.

    Returns
    =======
    bcast_shapes : list of ndarray
        The broadcasted values of `params`.
    """
    max_fn = maximum if use_theano else max

    rev_extra_dims = []
    for ndim_param, param_shape in zip(ndims_params, param_shapes):
        # We need this in order to use `len`
        param_shape = tuple(param_shape)
        extras = tuple(param_shape[: (len(param_shape) - ndim_param)])
        rev_extra_dims = [
            max_fn(a, b)
            for a, b in zip_longest(reversed(extras), rev_extra_dims, fillvalue=1)
        ]

    extra_dims = tuple(reversed(rev_extra_dims))

    bcast_shapes = [
        (extra_dims + tuple(param_shape)[-ndim_param:])
        if ndim_param > 0
        else extra_dims
        for ndim_param, param_shape in zip(ndims_params, param_shapes)
    ]

    return bcast_shapes


def broadcast_params(params, ndims_params):
    """Broadcast parameters that have different dimensions.

    >>> ndims_params = [1, 2]
    >>> mean = np.array([1, 2, 3])
    >>> cov = np.stack([np.eye(3), np.eye(3)])
    >>> params = [mean, cov]
    >>> res = broadcast_params(params, ndims_params)
    [array([[1, 2, 3]]),
    array([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]],
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])]

    Parameters
    ==========
    params : list of ndarray
        The parameters to broadcast.
    ndims_params : list of int
        The expected number of dimensions for each element in `params`.

    Returns
    =======
    bcast_params : list of ndarray
        The broadcasted values of `params`.
    """
    use_theano = False
    param_shapes = []
    for p in params:
        param_shape = p.shape
        use_theano |= isinstance(p, Variable)
        param_shapes.append(param_shape)

    shapes = params_broadcast_shapes(param_shapes, ndims_params, use_theano=use_theano)
    broadcast_to_fn = broadcast_to if use_theano else np.broadcast_to

    bcast_params = [
        broadcast_to_fn(param, shape) for shape, param in zip(shapes, params)
    ]

    return bcast_params


class RandomStream:
    """Module component with similar interface to `numpy.random.RandomState`.

    Attributes
    ----------
    seed: None or int
        A default seed to initialize the RandomState instances after build.
    state_updates: list
        A list of pairs of the form `(input_r, output_r)`.  This will be
        over-ridden by the module instance to contain stream generators.
    default_instance_seed: int
        Instance variable should take None or integer value. Used to seed the
        random number generator that provides seeds for member streams.
    gen_seedgen: numpy.random.RandomState
        `RandomState` instance that `RandomStream.gen` uses to seed new
        streams.

    """

    def __init__(self, seed=None, namespace=None):
        if namespace is None:
            from theano.tensor.random import basic  # pylint: disable=import-self

            self.namespaces = [basic]
        else:
            self.namespaces = [namespace]

        self.default_instance_seed = seed
        self.state_updates = []
        self.gen_seedgen = np.random.RandomState(seed)

    def __getattr__(self, obj):

        ns_obj = next(
            (getattr(ns, obj) for ns in self.namespaces if hasattr(ns, obj)), None
        )

        if ns_obj is None:
            raise AttributeError("No attribute {}.".format(obj))

        from theano.tensor.random.op import RandomVariable

        if isinstance(ns_obj, RandomVariable):

            @wraps(ns_obj)
            def meta_obj(*args, **kwargs):
                return self.gen(ns_obj, *args, **kwargs)

        else:
            raise AttributeError("No attribute {}.".format(obj))

        setattr(self, obj, meta_obj)
        return getattr(self, obj)

    def updates(self):
        return list(self.state_updates)

    def seed(self, seed=None):
        """
        Re-initialize each random stream.

        Parameters
        ----------
        seed : None or integer in range 0 to 2**30
            Each random stream will be assigned a unique state that depends
            deterministically on this value.

        Returns
        -------
        None

        """
        if seed is None:
            seed = self.default_instance_seed

        self.gen_seedgen.seed(seed)

        for old_r, new_r in self.state_updates:
            old_r_seed = self.gen_seedgen.randint(2 ** 30)
            old_r.set_value(np.random.RandomState(int(old_r_seed)), borrow=True)

    def gen(self, op, *args, **kwargs):
        """Create a new random stream in this container.

        Parameters
        ----------
        op : RandomVariable
            A `RandomVariable` instance
        args
            Positional arguments passed to `op`.
        kwargs
            Keyword arguments passed to `op`.

        Returns
        -------
        TensorVariable
            The symbolic random draw part of op()'s return value.
            This function stores the updated `RandomStateType` variable
            for use at `build` time.

        """
        if "rng" in kwargs:
            raise TypeError(
                "The rng option cannot be used with a variate in a RandomStream"
            )

        # Generate a new random state
        seed = int(self.gen_seedgen.randint(2 ** 30))
        random_state_variable = shared(np.random.RandomState(seed))

        # Distinguish it from other shared variables (why?)
        random_state_variable.tag.is_rng = True

        # Generate the sample
        out = op(*args, **kwargs, rng=random_state_variable)
        out.rng = random_state_variable

        # Update the tracked states
        new_r = out.owner.outputs[0]
        out.update = (random_state_variable, new_r)
        self.state_updates.append(out.update)

        random_state_variable.default_update = new_r

        return out

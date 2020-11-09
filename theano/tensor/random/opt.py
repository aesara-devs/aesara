import theano.tensor as tt
from theano import config
from theano.compile import optdb
from theano.compile.ops import Shape
from theano.gof.op import compute_test_value
from theano.gof.opt import local_optimizer
from theano.tensor.elemwise import DimShuffle
from theano.tensor.extra_ops import broadcast_to
from theano.tensor.opt import in2out
from theano.tensor.random.op import RandomVariable
from theano.tensor.random.utils import broadcast_params


@local_optimizer([RandomVariable])
def random_make_inplace(fgraph, node):
    op = node.op

    if isinstance(op, RandomVariable) and not op.inplace:
        name, ndim_supp, ndims_params, dtype, _ = op._props()
        new_op = type(op)(name, ndim_supp, ndims_params, dtype, True)
        return new_op.make_node(*node.inputs).outputs

    return False


optdb.register(
    "random_make_inplace",
    in2out(random_make_inplace, ignore_newtrees=True),
    99,
    "fast_run",
    "inplace",
)


def lift_rv_shapes(node):
    """Lift `RandomVariable`'s shape-related parameters.

    In other words, this will broadcast the distribution parameters and
    extra dimensions added by the `size` parameter.

    For example, ``normal([0.0, 1.0], 5.0, size=(3, 2))`` becomes
    ``normal([[0., 1.], [0., 1.], [0., 1.]], [[5., 5.], [5., 5.], [5., 5.]])``.

    """

    if not isinstance(node.op, RandomVariable):
        return False

    rng, size, dtype, *dist_params = node.inputs

    dist_params = broadcast_params(dist_params, node.op.ndims_params)

    dist_params = [
        broadcast_to(
            p, (tuple(size) + tuple(p.shape)) if node.op.ndim_supp > 0 else size
        )
        for p in dist_params
    ]

    return node.op.make_node(rng, None, dtype, *dist_params)


@local_optimizer([DimShuffle])
def local_dimshuffle_rv_lift(fgraph, node):
    """Lift `DimShuffle`s through `RandomVariable` `Op`s.

    For example, ``normal(mu, std).T == normal(mu.T, std.T)``.

    The basic idea behind this optimization is that we need to separate the
    `DimShuffle`ing into independent `DimShuffle`s that each occur in two
    distinct sub-spaces: the parameters and ``size`` (i.e. replications)
    sub-spaces.

    If a `DimShuffle` exchanges dimensions across those two sub-spaces, then we
    don't do anything.

    Otherwise, if the `DimShuffle` only exchanges dimensions within each of
    those sub-spaces, we can break it apart and apply the parameter-space
    `DimShuffle` to the `RandomVariable`'s distribution parameters, and the
    apply the replications-space `DimShuffle` to the `RandomVariable`'s``size``
    tuple.  The latter is a particularly simple rearranging of a tuple, but the
    former requires a little more work.
    """

    ds_op = node.op

    if not isinstance(ds_op, DimShuffle):
        return False

    base_rv = node.inputs[0]
    rv_node = base_rv.owner

    if not (
        rv_node and isinstance(rv_node.op, RandomVariable) and rv_node.op.ndim_supp == 0
    ):
        return False

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if not all(
        (n == node or isinstance(n.op, Shape)) for n, i in fgraph.clients[base_rv]
    ):
        return False

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs

    # We need to know the dimensions that were *not* added by the `size`
    # parameter (i.e. the dimensions corresponding to independent variates with
    # different parameter values)
    num_ind_dims = None
    if len(dist_params) == 1:
        num_ind_dims = dist_params[0].ndim
    else:
        # When there is more than one distribution parameter, assume that all
        # of them will broadcast to the maximum number of dimensions
        num_ind_dims = max(d.ndim for d in dist_params)

    # If the indices in `ds_new_order` are entirely within the replication
    # indices group or the independent variates indices group, then we can apply
    # this optimization.

    ds_new_order = ds_op.new_order
    # Create a map from old index order to new/`DimShuffled` index order
    dim_orders = [(n, d) for n, d in enumerate(ds_new_order) if isinstance(d, int)]

    # Find the index at which the replications/independents split occurs
    reps_ind_split_idx = len(dim_orders) - (num_ind_dims + rv_op.ndim_supp)

    ds_reps_new_dims = dim_orders[:reps_ind_split_idx]
    ds_ind_new_dims = dim_orders[reps_ind_split_idx:]
    ds_only_in_ind = ds_ind_new_dims and all(
        d >= reps_ind_split_idx for n, d in ds_ind_new_dims
    )

    if ds_only_in_ind:

        # Update the `size` array to reflect the `DimShuffle`d dimensions,
        # since the trailing dimensions in `size` represent the independent
        # variates dimensions (for univariate distributions, at least)
        new_size = (
            [
                tt.constant(1, dtype="int64") if o == "x" else size[o]
                for o in ds_new_order
            ]
            if tt.get_vector_length(size) > 0
            else size
        )

        # Compute the new axes parameter(s) for the `DimShuffle` that will be
        # applied to the `RandomVariable` parameters (they need to be offset)
        rv_params_new_order = [
            d - reps_ind_split_idx if isinstance(d, int) else d
            for d in ds_new_order[ds_ind_new_dims[0][0] :]
        ]

        # Lift the `DimShuffle`s into the parameters
        # NOTE: The parameters might not be broadcasted against each other, so
        # we can only apply the parts of the `DimShuffle` that are relevant.
        new_dist_params = []
        for d in dist_params:
            if d.ndim < len(ds_ind_new_dims):
                _rv_params_new_order = [
                    o
                    for o in rv_params_new_order
                    if (isinstance(o, int) and o < d.ndim) or o == "x"
                ]
            else:
                _rv_params_new_order = rv_params_new_order

            new_dist_params.append(
                type(ds_op)(d.type.broadcastable, _rv_params_new_order)(d)
            )
        new_node = rv_op.make_node(rng, new_size, dtype, *new_dist_params)

        if config.compute_test_value != "off":
            compute_test_value(new_node)

        return [new_node.outputs[1]]

    ds_only_in_reps = ds_reps_new_dims and all(
        d < reps_ind_split_idx for n, d in ds_reps_new_dims
    )

    if ds_only_in_reps:
        # Update the `size` array to reflect the `DimShuffle`d dimensions.
        # There should be no need to `DimShuffle` now.
        new_size = [
            tt.constant(1, dtype="int64") if o == "x" else size[o] for o in ds_new_order
        ]

        new_node = rv_op.make_node(rng, new_size, dtype, *dist_params)

        if config.compute_test_value != "off":
            compute_test_value(new_node)

        return [new_node.outputs[1]]

    return False

from theano.compile import optdb
from theano.gof.opt import local_optimizer
from theano.tensor.opt import in2out
from theano.tensor.random.op import RandomVariable


@local_optimizer([RandomVariable])
def random_make_inplace(fgraph, node):
    op = node.op

    if isinstance(op, RandomVariable) and not op.inplace:

        name, ndim_supp, ndims_params, dtype, _ = op._props()
        new_op = type(op)(name, ndim_supp, ndims_params, dtype, True)
        # rng, size, dtype, *dist_params = node.inputs

        return new_op.make_node(*node.inputs).outputs

    return False


optdb.register(
    "random_make_inplace",
    in2out(random_make_inplace, ignore_newtrees=True),
    99,
    "fast_run",
    "inplace",
)

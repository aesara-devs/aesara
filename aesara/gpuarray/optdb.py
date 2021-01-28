from aesara.compile import optdb
from aesara.graph.opt import GraphToGPULocalOptGroup, TopoOptimizer, local_optimizer
from aesara.graph.optdb import DB, EquilibriumDB, LocalGroupDB, SequenceDB


gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()

# Not used for an EquilibriumOptimizer. It has the "tracks" that we need for GraphToGPUDB.
gpu_optimizer2 = EquilibriumDB()

gpu_seqopt = SequenceDB()

# do not add 'fast_run' to these two as this would always enable gpuarray mode
optdb.register(
    "gpuarray_opt",
    gpu_seqopt,
    optdb.__position__.get("add_destroy_handler", 49.5) - 1,
    "gpuarray",
)


pool_db = LocalGroupDB()
pool_db2 = LocalGroupDB(local_opt=GraphToGPULocalOptGroup)
pool_db2.__name__ = "pool_db2"

matrix_ops_db = LocalGroupDB()
matrix_ops_db2 = LocalGroupDB(local_opt=GraphToGPULocalOptGroup)
matrix_ops_db2.__name__ = "matrix_ops_db2"

abstract_batch_norm_db = LocalGroupDB()
abstract_batch_norm_db2 = LocalGroupDB(local_opt=GraphToGPULocalOptGroup)
abstract_batch_norm_db2.__name__ = "abstract_batch_norm_db2"

abstract_batch_norm_groupopt = LocalGroupDB()
abstract_batch_norm_groupopt.__name__ = "gpuarray_batchnorm_opts"


def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop("name")) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, "fast_run", "gpuarray", *tags)
        return local_opt

    return f


def register_opt2(tracks, *tags, **kwargs):
    """
    Decorator for the new GraphToGPU optimizer.
    Takes an extra parameter(Op) compared to register_opt decorator.

    Parameters
    ----------
    tracks : List of Op class Or Op instance or None
        The Node's Op to which optimization is being applied.

    tags : String
        The optimization tag to which the optimizer will be registered.

    """

    def f(local_opt):
        name = (kwargs and kwargs.pop("name")) or local_opt.__name__
        if isinstance(local_opt, DB):
            opt = local_opt
        else:
            opt = local_optimizer(tracks)(local_opt)
        gpu_optimizer2.register(name, opt, "fast_run", "gpuarray", *tags)
        return local_opt

    return f


def register_inplace(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop("name")) or local_opt.__name__
        optdb.register(
            name,
            TopoOptimizer(local_opt, failure_callback=TopoOptimizer.warn_inplace),
            60,
            "fast_run",
            "inplace",
            "gpuarray",
            *tags,
        )
        return local_opt

    return f


# Register GPU convolution implementation
# They are tried in a specific order so we can control
# which ones take precedence over others.
abstractconv_groupopt = LocalGroupDB()
abstractconv_groupopt.__name__ = "gpuarray_abstractconv_opts"
register_opt("fast_compile")(abstractconv_groupopt)


class GraphToGPUDB(DB):
    """
    Retrieves the list local optimizers based on the optimizer flag's value
    from EquilibriumOptimizer by calling the method query.

    """

    def query(self, *tags, **kwtags):
        from aesara.gpuarray.opt import GraphToGPU

        opt = gpu_optimizer2.query(*tags, **kwtags)
        return GraphToGPU(opt.local_optimizers_all, opt.local_optimizers_map)

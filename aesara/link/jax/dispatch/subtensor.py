import jax

from aesara.link.jax.dispatch.basic import jax_funcify
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)
from aesara.tensor.type_other import MakeSlice


@jax_funcify.register(Subtensor)
@jax_funcify.register(AdvancedSubtensor)
@jax_funcify.register(AdvancedSubtensor1)
def jax_funcify_Subtensor(op, **kwargs):

    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):

        indices = indices_from_subtensor(ilists, idx_list)

        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor


@jax_funcify.register(IncSubtensor)
@jax_funcify.register(AdvancedIncSubtensor1)
def jax_funcify_IncSubtensor(op, **kwargs):

    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):
        jax_fn = getattr(jax.ops, "index_update", None)

        if jax_fn is None:

            def jax_fn(x, indices, y):
                return x.at[indices].set(y)

    else:
        jax_fn = getattr(jax.ops, "index_add", None)

        if jax_fn is None:

            def jax_fn(x, indices, y):
                return x.at[indices].add(y)

    def incsubtensor(x, y, *ilist, jax_fn=jax_fn, idx_list=idx_list):
        indices = indices_from_subtensor(ilist, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return jax_fn(x, indices, y)

    return incsubtensor


@jax_funcify.register(AdvancedIncSubtensor)
def jax_funcify_AdvancedIncSubtensor(op, **kwargs):

    if getattr(op, "set_instead_of_inc", False):
        jax_fn = getattr(jax.ops, "index_update", None)

        if jax_fn is None:

            def jax_fn(x, indices, y):
                return x.at[indices].set(y)

    else:
        jax_fn = getattr(jax.ops, "index_add", None)

        if jax_fn is None:

            def jax_fn(x, indices, y):
                return x.at[indices].add(y)

    def advancedincsubtensor(x, y, *ilist, jax_fn=jax_fn):
        return jax_fn(x, ilist, y)

    return advancedincsubtensor


@jax_funcify.register(MakeSlice)
def jax_funcify_MakeSlice(op, **kwargs):
    def makeslice(*x):
        return slice(*x)

    return makeslice

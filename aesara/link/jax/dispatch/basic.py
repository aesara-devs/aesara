import warnings
from functools import singledispatch

import jax
import jax.numpy as jnp
import numpy as np

from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.configdefaults import config
from aesara.graph.fg import FunctionGraph
from aesara.ifelse import IfElse
from aesara.link.utils import fgraph_to_python
from aesara.raise_op import CheckAndRaise


if config.floatX == "float64":
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)

# XXX: Enabling this will break some shape-based functionality, and severely
# limit the types of graphs that can be converted.
# See https://github.com/google/jax/blob/4d556837cc9003492f674c012689efc3d68fdf5f/design_notes/omnistaging.md
# Older versions < 0.2.0 do not have this flag so we don't need to set it.
try:
    jax.config.disable_omnistaging()
except AttributeError:
    pass
except Exception as e:
    # The version might be >= 0.2.12, which means that omnistaging can't be
    # disabled
    warnings.warn(f"JAX omnistaging couldn't be disabled: {e}")


@singledispatch
def jax_typify(data, dtype=None, **kwargs):
    r"""Convert instances of Aesara `Type`\s to JAX types."""
    if dtype is None:
        return data
    else:
        return jnp.array(data, dtype=dtype)


@jax_typify.register(np.ndarray)
def jax_typify_ndarray(data, dtype=None, **kwargs):
    return jnp.array(data, dtype=dtype)


@singledispatch
def jax_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a JAX compatible function from an Aesara `Op`."""
    raise NotImplementedError(f"No JAX conversion for the given `Op`: {op}")


@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="jax_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        jax_funcify,
        type_conversion_fn=jax_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@jax_funcify.register(IfElse)
def jax_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = jax.lax.cond(
            cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
        )
        return res if n_outs > 1 else res[0]

    return ifelse


@jax_funcify.register(CheckAndRaise)
def jax_funcify_CheckAndRaise(op, **kwargs):

    raise NotImplementedError(
        f"""This exception is raised because you tried to convert an aesara graph with a `CheckAndRaise` Op (message: {op.msg}) to JAX.

        JAX uses tracing to jit-compile functions, and assertions typically
        don't do well with tracing. The appropriate workaround depends on what
        you intended to do with the assertions in the first place.

        Note that all assertions can be removed from the graph by adding
        `local_remove_all_assert` to the rewrites."""
    )


def jnp_safe_copy(x):
    try:
        res = jnp.copy(x)
    except NotImplementedError:
        warnings.warn(
            "`jnp.copy` is not implemented yet. " "Using the object's `copy` method."
        )
        if hasattr(x, "copy"):
            res = jnp.array(x.copy())
        else:
            warnings.warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@jax_funcify.register(DeepCopyOp)
def jax_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return jnp_safe_copy(x)

    return deepcopyop


@jax_funcify.register(ViewOp)
def jax_funcify_ViewOp(op, **kwargs):
    def viewop(x):
        return x

    return viewop

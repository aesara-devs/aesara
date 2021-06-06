from numpy.random import Generator, RandomState

from aesara.graph.basic import Constant
from aesara.link.basic import JITLinker


class JAXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX."""

    def fgraph_convert(self, fgraph, **kwargs):
        from aesara.link.jax.dispatch import jax_funcify

        return jax_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        import jax

        # I suppose we can consider `Constant`s to be "static" according to
        # JAX.
        static_argnums = [
            n for n, i in enumerate(self.fgraph.inputs) if isinstance(i, Constant)
        ]
        return jax.jit(fn, static_argnums=static_argnums)

    def create_thunk_inputs(self, storage_map):
        from aesara.link.jax.dispatch import jax_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], (RandomState, Generator)):
                new_value = jax_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                # We need to remove the reference-based connection to the
                # original `RandomState`/shared variable's storage, because
                # subsequent attempts to use the same shared variable within
                # other non-JAXified graphs will have problems.
                sinput = [new_value]
            thunk_inputs.append(sinput)

        return thunk_inputs

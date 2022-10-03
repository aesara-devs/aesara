import warnings

from numpy.random import Generator, RandomState

from aesara.compile.sharedvalue import SharedVariable, shared
from aesara.graph.basic import Constant
from aesara.link.basic import JITLinker


class JAXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX."""

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from aesara.link.jax.dispatch import jax_funcify
        from aesara.tensor.random.type import RandomType

        shared_rng_inputs = [
            inp
            for inp in fgraph.inputs
            if (isinstance(inp, SharedVariable) and isinstance(inp.type, RandomType))
        ]

        # Replace any shared RNG inputs so that their values can be updated in place
        # without affecting the original RNG container. This is necessary because
        # JAX does not accept RandomState/Generators as inputs, and they will have to
        # be typyfied
        if shared_rng_inputs:
            warnings.warn(
                f"The RandomType SharedVariables {shared_rng_inputs} will not be used "
                f"in the compiled JAX graph. Instead a copy will be used.",
                UserWarning,
            )
            new_shared_rng_inputs = [
                shared(inp.get_value(borrow=False)) for inp in shared_rng_inputs
            ]

            fgraph.replace_all(
                zip(shared_rng_inputs, new_shared_rng_inputs),
                import_missing=True,
                reason="JAXLinker.fgraph_convert",
            )

            for old_inp, new_inp in zip(shared_rng_inputs, new_shared_rng_inputs):
                new_inp_storage = [new_inp.get_value(borrow=True)]
                storage_map[new_inp] = new_inp_storage
                old_inp_storage = storage_map.pop(old_inp)
                input_storage[input_storage.index(old_inp_storage)] = new_inp_storage
                fgraph.remove_input(
                    fgraph.inputs.index(old_inp), reason="JAXLinker.fgraph_convert"
                )

        return jax_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **kwargs
        )

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
                sinput[0] = new_value
            thunk_inputs.append(sinput)

        return thunk_inputs

import numba

from aesara.link.basic import JITLinker


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(
        self, fgraph, order, input_storage, output_storage, storage_map, **kwargs
    ):
        from aesara.link.numba.dispatch import numba_funcify

        return numba_funcify(
            fgraph, order, input_storage, output_storage, storage_map, **kwargs
        )

    def jit_compile(self, fn):
        jitted_fn = numba.njit(fn)
        return jitted_fn

    def create_thunk_inputs(self, storage_map):
        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            # TODO:When RandomVariable conversion is implemented
            # do RandomState typification over here.
            thunk_inputs.append(sinput)

        return thunk_inputs

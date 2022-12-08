from typing import TYPE_CHECKING, Any

import numpy as np

import aesara
from aesara.link.basic import JITLinker


if TYPE_CHECKING:
    from aesara.graph.basic import Variable


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def output_filter(self, var: "Variable", out: Any) -> Any:
        if not isinstance(var, np.ndarray) and isinstance(
            var.type, aesara.tensor.TensorType
        ):
            return var.type.filter(out, allow_downcast=True)

        return out

    def fgraph_convert(self, fgraph, **kwargs):
        from aesara.link.numba.dispatch import numba_funcify

        return numba_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        from aesara.link.numba.dispatch import numba_njit

        jitted_fn = numba_njit(fn)
        return jitted_fn

    def create_thunk_inputs(self, storage_map):
        from numpy.random import RandomState

        from aesara.link.numba.dispatch import numba_const_convert

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], RandomState):
                new_value = numba_const_convert(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                # We need to remove the reference-based connection to the
                # original `RandomState`/shared variable's storage, because
                # subsequent attempts to use the same shared variable within
                # other non-Numba-fied graphs will have problems.
                sinput = [new_value]
            thunk_inputs.append(sinput)

        return thunk_inputs

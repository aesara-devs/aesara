"""
Implementations of BLAS Ops based on scipy's BLAS bindings.
"""

import numpy as np

from aesara.graph.rewriting.basic import in2out
from aesara.tensor.blas import (
    Ger,
    blas_optdb,
    ger,
    ger_destructive,
    have_fblas,
    node_rewriter,
    optdb,
)


if have_fblas:
    from aesara.tensor.blas import fblas

    _blas_ger_fns = {
        np.dtype("float32"): fblas.sger,
        np.dtype("float64"): fblas.dger,
        np.dtype("complex64"): fblas.cgeru,
        np.dtype("complex128"): fblas.zgeru,
    }


class ScipyGer(Ger):
    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl == "py":
            node.tag.local_ger = _blas_ger_fns[np.dtype(node.inputs[0].type.dtype)]

    def perform(self, node, inputs, output_storage):
        cA, calpha, cx, cy = inputs
        (cZ,) = output_storage
        # N.B. some versions of scipy (e.g. mine) don't actually work
        # in-place on a, even when I tell it to.
        A = cA
        local_ger = node.tag.local_ger
        if A.size == 0:
            # We don't have to compute anything, A is empty.
            # We need this special case because Numpy considers it
            # C-contiguous, which is confusing.
            if not self.destructive:
                # Sometimes numpy thinks empty matrices can share memory,
                # so here to stop DebugMode from complaining.
                A = A.copy()
        elif A.flags["C_CONTIGUOUS"]:
            A = local_ger(calpha, cy, cx, a=A.T, overwrite_a=int(self.destructive)).T
        else:
            A = local_ger(calpha, cx, cy, a=A, overwrite_a=int(self.destructive))
        cZ[0] = A


scipy_ger_no_inplace = ScipyGer(False)
scipy_ger_inplace = ScipyGer(True)


@node_rewriter([ger, ger_destructive])
def use_scipy_ger(fgraph, node):
    if node.op == ger:
        return [scipy_ger_no_inplace(*node.inputs)]


@node_rewriter([scipy_ger_no_inplace])
def make_ger_destructive(fgraph, node):
    if node.op == scipy_ger_no_inplace:
        return [scipy_ger_inplace(*node.inputs)]


use_scipy_blas = in2out(use_scipy_ger)
make_scipy_blas_destructive = in2out(make_ger_destructive)

if have_fblas:
    # scipy_blas is scheduled in the blas_optdb very late, because scipy sortof
    # sucks, but it is almost always present.
    # C implementations should be scheduled earlier than this, so that they take
    # precedence. Once the original Ger is replaced, then these optimizations
    # have no effect.
    blas_optdb.register("scipy_blas", use_scipy_blas, "fast_run", position=100)

    # this matches the InplaceBlasOpt defined in blas.py
    optdb.register(
        "make_scipy_blas_destructive",
        make_scipy_blas_destructive,
        "fast_run",
        "inplace",
        position=70.0,
    )

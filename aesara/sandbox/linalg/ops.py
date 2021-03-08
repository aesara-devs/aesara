import logging

import aesara.tensor
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.graph.opt import GlobalOptimizer, local_optimizer
from aesara.tensor import basic as aet
from aesara.tensor.basic_opt import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from aesara.tensor.blas import Dot22
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import Dot, Prod, dot, log
from aesara.tensor.math import pow as aet_pow
from aesara.tensor.math import prod
from aesara.tensor.nlinalg import MatrixInverse, det, matrix_inverse, trace
from aesara.tensor.slinalg import Cholesky, Solve, cholesky, imported_scipy, solve


logger = logging.getLogger(__name__)


class Hint(Op):
    """
    Provide arbitrary information to the optimizer.

    These ops are removed from the graph during canonicalization
    in order to not interfere with other optimizations.
    The idea is that prior to canonicalization, one or more Features of the
    fgraph should register the information contained in any Hint node, and
    transfer that information out of the graph.

    """

    __props__ = ("hints",)

    def __init__(self, **kwargs):
        self.hints = tuple(kwargs.items())
        self.view_map = {0: [0]}

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outstor):
        outstor[0][0] = inputs[0]

    def grad(self, inputs, g_out):
        return g_out


def hints(variable):
    if variable.owner and isinstance(variable.owner.op, Hint):
        return dict(variable.owner.op.hints)
    else:
        return {}


@register_canonicalize
@local_optimizer([Hint])
def remove_hint_nodes(fgraph, node):
    if isinstance(node, Hint):
        # transfer hints from graph to Feature
        try:
            for k, v in node.op.hints:
                fgraph.hints_feature.add_hint(node.inputs[0], k, v)
        except AttributeError:
            pass
        return node.inputs


class HintsFeature:
    """
    FunctionGraph Feature to track matrix properties.

    This is a similar feature to variable 'tags'. In fact, tags are one way
    to provide hints.

    This class exists because tags were not documented well, and the
    semantics of how tag information should be moved around during
    optimizations was never clearly spelled out.

    Hints are assumptions about mathematical properties of variables.
    If one variable is substituted for another by an optimization,
    then it means that the assumptions should be transferred to the
    new variable.

    Hints are attached to 'positions in a graph' rather than to variables
    in particular, although Hints are originally attached to a particular
    positition in a graph *via* a variable in that original graph.

    Examples of hints are:
    - shape information
    - matrix properties (e.g. symmetry, psd, banded, diagonal)

    Hint information is propagated through the graph similarly to graph
    optimizations, except that adding a hint does not change the graph.
    Adding a hint is not something that debugmode will check.

    #TODO: should a Hint be an object that can actually evaluate its
    #      truthfulness?
    #      Should the PSD property be an object that can check the
    #      PSD-ness of a variable?

    """

    def add_hint(self, r, k, v):
        logger.debug(f"adding hint; {r}, {k}, {v}")
        self.hints[r][k] = v

    def ensure_init_r(self, r):
        if r not in self.hints:
            self.hints[r] = {}

    #
    #
    # Feature inteface
    #
    #
    def on_attach(self, fgraph):
        assert not hasattr(fgraph, "hints_feature")
        fgraph.hints_feature = self
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.hints = {}
        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.hints:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.hints
            return

        for i, r in enumerate(node.inputs + node.outputs):
            # make sure we have shapes for the inputs
            self.ensure_init_r(r)

    def update_second_from_first(self, r0, r1):
        old_hints = self.hints[r0]
        new_hints = self.hints[r1]
        for k, v in old_hints.items():
            if k in new_hints and new_hints[k] is not v:
                raise NotImplementedError()
            if k not in new_hints:
                new_hints[k] = v

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # TODO:
        # This tells us that r and new_r must have the same shape
        # if we didn't know that the shapes are related, now we do.
        self.ensure_init_r(new_r)
        self.update_second_from_first(r, new_r)
        self.update_second_from_first(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.


class HintsOptimizer(GlobalOptimizer):
    """
    Optimizer that serves to add HintsFeature as an fgraph feature.
    """

    def __init__(self):
        super().__init__()

    def add_requirements(self, fgraph):
        fgraph.attach_feature(HintsFeature())

    def apply(self, fgraph):
        pass


# -1 should make it run right before the first merge
aesara.compile.mode.optdb.register(
    "HintsOpt", HintsOptimizer(), -1, "fast_run", "fast_compile"
)


def psd(v):
    r"""
    Apply a hint that the variable `v` is positive semi-definite, i.e.
    it is a symmetric matrix and :math:`x^T A x \ge 0` for any vector x.

    """
    return Hint(psd=True, symmetric=True)(v)


def is_psd(v):
    return hints(v).get("psd", False)


def is_symmetric(v):
    return hints(v).get("symmetric", False)


def is_positive(v):
    if hints(v).get("positive", False):
        return True
    # TODO: how to handle this - a registry?
    #      infer_hints on Ops?
    logger.debug(f"is_positive: {v}")
    if v.owner and v.owner.op == aet_pow:
        try:
            exponent = aet.get_scalar_constant_value(v.owner.inputs[1])
        except NotScalarConstantError:
            return False
        if 0 == exponent % 2:
            return True
    return False


@register_canonicalize
@local_optimizer([DimShuffle])
def transinv_to_invtrans(fgraph, node):
    if isinstance(node.op, DimShuffle):
        if node.op.new_order == (1, 0):
            (A,) = node.inputs
            if A.owner:
                if isinstance(A.owner.op, MatrixInverse):
                    (X,) = A.owner.inputs
                    return [A.owner.op(node.op(X))]


@register_stabilize
@local_optimizer([Dot, Dot22])
def inv_as_solve(fgraph, node):
    if not imported_scipy:
        return False
    if isinstance(node.op, (Dot, Dot22)):
        l, r = node.inputs
        if l.owner and l.owner.op == matrix_inverse:
            return [solve(l.owner.inputs[0], r)]
        if r.owner and r.owner.op == matrix_inverse:
            if is_symmetric(r.owner.inputs[0]):
                return [solve(r.owner.inputs[0], l.T).T]
            else:
                return [solve(r.owner.inputs[0].T, l.T).T]


@register_stabilize
@register_canonicalize
@local_optimizer([Solve])
def tag_solve_triangular(fgraph, node):
    """
    If a general solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.

    """
    if node.op == solve:
        if node.op.A_structure == "general":
            A, b = node.inputs  # result is solution Ax=b
            if A.owner and isinstance(A.owner.op, type(cholesky)):
                if A.owner.op.lower:
                    return [Solve("lower_triangular")(A, b)]
                else:
                    return [Solve("upper_triangular")(A, b)]
            if (
                A.owner
                and isinstance(A.owner.op, DimShuffle)
                and A.owner.op.new_order == (1, 0)
            ):
                (A_T,) = A.owner.inputs
                if A_T.owner and isinstance(A_T.owner.op, type(cholesky)):
                    if A_T.owner.op.lower:
                        return [Solve("upper_triangular")(A, b)]
                    else:
                        return [Solve("lower_triangular")(A, b)]


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([DimShuffle])
def no_transpose_symmetric(fgraph, node):
    if isinstance(node.op, DimShuffle):
        x = node.inputs[0]
        if x.type.ndim == 2 and is_symmetric(x):
            # print 'UNDOING TRANSPOSE', is_symmetric(x), x.ndim
            if node.op.new_order == [1, 0]:
                return [x]


@register_stabilize
@local_optimizer(None)  # XXX: solve is defined later and can't be used here
def psd_solve_with_chol(fgraph, node):
    if node.op == solve:
        A, b = node.inputs  # result is solution Ax=b
        if is_psd(A):
            L = cholesky(A)
            # N.B. this can be further reduced to a yet-unwritten cho_solve Op
            #     __if__ no other Op makes use of the the L matrix during the
            #     stabilization
            Li_b = Solve("lower_triangular")(L, b)
            x = Solve("upper_triangular")(L.T, Li_b)
            return [x]


@register_stabilize
@register_specialize
@local_optimizer(None)  # XXX: det is defined later and can't be used here
def local_det_chol(fgraph, node):
    """
    If we have det(X) and there is already an L=cholesky(X)
    floating around, then we can use prod(diag(L)) to get the determinant.

    """
    if node.op == det:
        (x,) = node.inputs
        for (cl, xpos) in fgraph.clients[x]:
            if isinstance(cl.op, Cholesky):
                L = cl.outputs[0]
                return [prod(aet.extract_diag(L) ** 2)]


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([log])
def local_log_prod_sqr(fgraph, node):
    if node.op == log:
        (x,) = node.inputs
        if x.owner and isinstance(x.owner.op, Prod):
            # we cannot always make this substitution because
            # the prod might include negative terms
            p = x.owner.inputs[0]

            # p is the matrix we're reducing with prod
            if is_positive(p):
                return [log(p).sum(axis=x.owner.op.axis)]

            # TODO: have a reduction like prod and sum that simply
            #      returns the sign of the prod multiplication.


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([log])
def local_log_pow(fgraph, node):
    if node.op == log:
        (x,) = node.inputs
        if x.owner and x.owner.op == aet_pow:
            base, exponent = x.owner.inputs
            # TODO: reason to be careful with dtypes?
            return [exponent * log(base)]


def spectral_radius_bound(X, log2_exponent):
    """
    Returns upper bound on the largest eigenvalue of square symmetrix matrix X.

    log2_exponent must be a positive-valued integer. The larger it is, the
    slower and tighter the bound. Values up to 5 should usually suffice. The
    algorithm works by multiplying X by itself this many times.

    From V.Pan, 1990. "Estimating the Extremal Eigenvalues of a Symmetric
    Matrix", Computers Math Applic. Vol 20 n. 2 pp 17-22.
    Rq: an efficient algorithm, not used here, is defined in this paper.

    """
    if X.type.ndim != 2:
        raise TypeError("spectral_radius_bound requires a matrix argument", X)
    if not isinstance(log2_exponent, int):
        raise TypeError(
            "spectral_radius_bound requires an integer exponent", log2_exponent
        )
    if log2_exponent <= 0:
        raise ValueError(
            "spectral_radius_bound requires a strictly positive " "exponent",
            log2_exponent,
        )

    XX = X
    for i in range(log2_exponent):
        XX = dot(XX, XX)
    return aet_pow(trace(XX), 2 ** (-log2_exponent))

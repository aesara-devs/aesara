"""This module defines the base classes for graph rewriting."""
import abc
import copy
import functools
import inspect
import logging
import pdb
import sys
import time
import traceback
import warnings
from collections import UserList, defaultdict, deque
from collections.abc import Iterable
from functools import _compose_mro, partial, reduce  # type: ignore
from itertools import chain
from typing import TYPE_CHECKING, Callable, Dict
from typing import Iterable as IterableType
from typing import List, Optional, Sequence, Tuple, Union, cast

from typing_extensions import Literal

import aesara
from aesara.configdefaults import config
from aesara.graph import destroyhandler as dh
from aesara.graph.basic import (
    Apply,
    AtomicVariable,
    Constant,
    Variable,
    applys_between,
    io_toposort,
    vars_between,
)
from aesara.graph.features import AlreadyThere, Feature, NodeFinder
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.utils import AssocList, InconsistencyError
from aesara.misc.ordered_set import OrderedSet
from aesara.utils import flatten


if TYPE_CHECKING:
    from aesara.graph.rewriting.unify import Var


_logger = logging.getLogger("aesara.graph.rewriting.basic")

RemoveKeyType = Literal["remove"]
TransformOutputType = Union[
    bool,
    Sequence[Variable],
    Dict[Union[Variable, Literal["remove"]], Union[Variable, Sequence[Variable]]],
]
FailureCallbackType = Callable[
    [
        Exception,
        "NodeProcessingGraphRewriter",
        List[Tuple[Variable, None]],
        "NodeRewriter",
        Apply,
    ],
    None,
]


class MetaNodeRewriterSkip(AssertionError):
    """This is an `AssertionError`, but instead of having the
    `MetaNodeRewriter` print the error, it just skip that
    compilation.

    """


class Rewriter(abc.ABC):
    """Abstract base class for graph/term rewriters."""

    name: Optional[str] = None

    @abc.abstractmethod
    def add_requirements(self, fgraph: FunctionGraph):
        r"""Add `Feature`\s and other requirements to a `FunctionGraph`."""

    @abc.abstractmethod
    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        """Print a single-line, indented representation of the rewriter."""

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class GraphRewriter(Rewriter):
    """A rewriter that can be applied to a `FunctionGraph` in order to transform it.

    This class represents a generalized rewrite that includes the way a graph
    is traversed and/or changed as a whole.

    """

    @abc.abstractmethod
    def apply(self, fgraph):
        """Apply the rewriter to a `FunctionGraph`.

        It may use all the methods defined by the `FunctionGraph`. If the
        `GraphRewriter` needs to use a certain tool, such as an
        `InstanceFinder`, it can do so in its `add_requirements` method.

        """
        raise NotImplementedError()

    def optimize(self, *args, **kwargs):
        warnings.warn(
            "`GraphRewriter.optimize` is deprecated; use `GraphRewriter.rewrite` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rewrite(*args, **kwargs)

    def rewrite(self, fgraph, *args, **kwargs):
        """

        This is meant as a shortcut for the following::

            self.add_requirements(fgraph)
            self.apply(fgraph)

        """
        self.add_requirements(fgraph)
        return self.apply(fgraph, *args, **kwargs)

    def __call__(self, fgraph):
        """Rewrite a `FunctionGraph`."""
        return self.rewrite(fgraph)

    def add_requirements(self, fgraph):
        ...

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}",
            file=stream,
        )

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        if prof is not None:
            raise NotImplementedError(
                "The function `print_profile` must be overridden when the"
                " rewriter returns profiling information."
            )


class NodeRewriter(Rewriter):
    """A `Rewriter` that is applied to an `Apply` node."""

    def tracks(self) -> Optional[Sequence[Op]]:
        """Return the list of `Op` classes to which this rewrite applies.

        Returns ``None`` when the rewrite applies to all nodes.

        """
        return None

    @abc.abstractmethod
    def transform(
        self, fgraph: FunctionGraph, node: Apply, *args, **kwargs
    ) -> TransformOutputType:
        r"""Rewrite the sub-graph given by `node`.

        Subclasses should implement this function so that it returns one of the
        following:

        - ``False`` to indicate that this rewrite cannot be applied to `node`
        - A list of `Variable`\s to use in place of the `node`'s current outputs
        - A ``dict`` mapping old `Variable`\s to `Variable`\s, or the key
        ``"remove"`` mapping to a list of `Variable`\s to be removed.

        Parameters
        ----------
        fgraph
            A `FunctionGraph` containing `node`.
        node
            An `Apply` node to be rewritten.

        """

        raise NotImplementedError()

    def add_requirements(self, fgraph: FunctionGraph):
        r"""Add required `Feature`\s to `fgraph`."""

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)


class FromFunctionGraphRewriter(GraphRewriter):
    """A `GraphRewriter` constructed from a given function."""

    def __init__(self, fn, requirements=()):
        self.fn = fn
        self.requirements = requirements

    def apply(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.apply} id={id(self)}", file=stream)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __str__(self):
        return self.__name__


def graph_rewriter(f):
    """Decorator for `FromFunctionGraphRewriter`."""
    rval = FromFunctionGraphRewriter(f)
    rval.__name__ = f.__name__
    return rval


def inplace_graph_rewriter(f):
    """Decorator for `FromFunctionGraphRewriter` that also adds the `DestroyHandler` features."""
    dh_handler = dh.DestroyHandler
    requirements = (lambda fgraph: fgraph.attach_feature(dh_handler()),)
    rval = FromFunctionGraphRewriter(f, requirements)
    rval.__name__ = f.__name__
    return rval


class SequentialGraphRewriter(GraphRewriter, UserList):
    """A `GraphRewriter` that applies a list of rewriters sequentially."""

    @classmethod
    def warn(cls, exc, self, rewriter):
        """Default failure callback function for `SequentialGraphRewriter`."""
        _logger.error(f"{cls.__name__} apply {rewriter}")
        _logger.error("Traceback:")
        _logger.error(traceback.format_exc())
        if config.on_opt_error == "raise":
            raise exc
        elif config.on_opt_error == "pdb":
            pdb.post_mortem(sys.exc_info()[2])

    def __init__(self, *rewrites, failure_callback=None):
        """
        Parameters
        ----------
        *rewrites
            The List of rewriters to be applied to a node
        failure_callback
            A callback used when a failure happens during rewriting.

        """
        if len(rewrites) == 1 and isinstance(rewrites[0], (list, tuple)):
            rewrites = rewrites[0]

        super().__init__(rewrites)

        self.failure_callback = failure_callback

    def apply(self, fgraph):
        """Applies each `GraphRewriter` in ``self.data`` to `fgraph`."""
        l = []
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            sub_validate_time = [validate_before]
            callbacks_before = fgraph.execute_callbacks_times.copy()
        else:
            sub_validate_time = []
            callbacks_before = []
        callback_before = fgraph.execute_callbacks_time
        nb_node_before = len(fgraph.apply_nodes)
        sub_profs = []
        nb_nodes = []

        self.pre_profile = (
            self,
            l,
            -1,
            -1,
            nb_node_before,
            -1,
            sub_profs,
            sub_validate_time,
            nb_nodes,
            {},
        )
        try:
            for rewriter in self.data:
                try:
                    nb_nodes_before = len(fgraph.apply_nodes)
                    t0 = time.time()
                    sub_prof = rewriter.apply(fgraph)
                    l.append(float(time.time() - t0))
                    sub_profs.append(sub_prof)
                    nb_nodes.append((nb_nodes_before, len(fgraph.apply_nodes)))
                    if fgraph.profile:
                        sub_validate_time.append(fgraph.profile.validate_time)
                except AssertionError:
                    # do not catch Assertion failures
                    raise
                except Exception as e:
                    if self.failure_callback:
                        self.failure_callback(e, self, rewriter)
                        continue
                    else:
                        raise
        finally:

            if fgraph.profile:
                validate_time = fgraph.profile.validate_time - validate_before
                callbacks_time = {}
                for k, v in fgraph.execute_callbacks_times.items():
                    if k in callbacks_before:
                        t = v - callbacks_before[k]
                        if t > 0:
                            callbacks_time[k] = t
                    else:
                        callbacks_time[k] = v
            else:
                validate_time = None
                callbacks_time = {}
            callback_time = fgraph.execute_callbacks_time - callback_before
            self.pre_profile = (
                self,
                l,
                validate_time,
                callback_time,
                nb_node_before,
                len(fgraph.apply_nodes),
                sub_profs,
                sub_validate_time,
                nb_nodes,
                callbacks_time,
            )
        return self.pre_profile

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def add_requirements(self, fgraph):
        for rewrite in self.data:
            rewrite.add_requirements(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}", file=stream
        )
        # This way, -1 will do all depth
        if depth != 0:
            depth -= 1
            for rewrite in self.data:
                rewrite.print_summary(stream, level=(level + 2), depth=depth)

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        (
            rewrites,
            prof,
            validate_time,
            callback_time,
            nb_node_before,
            nb_node_after,
            sub_profs,
            sub_validate_time,
            nb_nodes,
            callbacks_time,
        ) = prof

        validate_time = validate_time or float("nan")
        callback_time = callback_time or float("nan")

        blanc = "    " * level

        print(blanc, cls.__name__, end=" ", file=stream)
        if hasattr(rewrites, "name"):
            print(blanc, rewrites.name, end=" ", file=stream)
        elif hasattr(rewrites, "__name__"):
            print(blanc, rewrites.__name__, end=" ", file=stream)
        print(
            (
                f" time {sum(prof):.3f}s for {int(nb_node_before)}/{int(nb_node_after)} nodes"
                " before/after rewriting"
            ),
            file=stream,
        )
        print(blanc, f"  {callback_time:.3f}s for callback", file=stream)
        print(blanc, f"      {validate_time:.3f}s for fgraph.validate()", file=stream)
        if callback_time > 1:
            print(blanc, "  callbacks_time", file=stream)
            for i in sorted(callbacks_time.items(), key=lambda a: -a[1]):
                if i[1] > 0:
                    # We want to have the __str__ called, so we can't
                    # just print i.
                    print(blanc, "      ", i[0], ",", i[1], file=stream)

        if level == 0:
            print(
                blanc,
                "  time      - (name, class, index, nodes before, nodes after) - validate time",
                file=stream,
            )
        ll = []
        for (rewrite, nb_n) in zip(rewrites, nb_nodes):
            if hasattr(rewrite, "__name__"):
                name = rewrite.__name__
            else:
                name = rewrite.name
            idx = rewrites.index(rewrite)
            ll.append((name, rewrite.__class__.__name__, idx) + nb_n)
        lll = sorted(zip(prof, ll), key=lambda a: a[0])

        for (t, rewrite) in lll[::-1]:
            i = rewrite[2]
            if sub_validate_time:
                val_time = sub_validate_time[i + 1] - sub_validate_time[i]
                print(
                    blanc,
                    f"  {t:.6f}s - {rewrite} - {val_time:.3f}s",
                    file=stream,
                )
            else:
                print(blanc, f"  {t:.6f}s - {rewrite}", file=stream)

            if sub_profs[i]:
                rewrites[i].print_profile(stream, sub_profs[i], level=level + 1)
        print(file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        """Merge two profiles."""
        new_t = []  # the times for the rewrites
        new_l = []  # the rewrites
        new_sub_profile = []
        # Merge common (i.e. same object) rewrites
        for l in set(prof1[0]).intersection(set(prof2[0])):
            idx1 = prof1[0].index(l)
            idx2 = prof2[0].index(l)
            new_t.append(prof1[1][idx1] + prof2[1][idx2])
            new_l.append(l)
            if hasattr(l, "merge_profile"):
                assert len(prof1[6][idx1]) == len(prof2[6][idx2])
                new_sub_profile.append(l.merge_profile(prof1[6][idx1], prof2[6][idx2]))
            else:
                new_sub_profile.append(None)

        from io import StringIO

        for l in set(prof1[0]).symmetric_difference(set(prof2[0])):
            # The set trick above only works for the same rewrite objects; it
            # doesn't work for equivalent rewrites, so we try to merge
            # equivalent rewrites here.
            new_l_names = [o.name for o in new_l]
            if l.name in new_l_names:
                idx = new_l_names.index(l.name)
                io1 = StringIO()
                io2 = StringIO()
                l.print_summary(io1)
                new_l[idx].print_summary(io2)
                if io1.read() == io2.read():
                    if l in prof1[0]:
                        p = prof1
                    else:
                        p = prof2
                    new_t[idx] += p[1][p[0].index(l)]
                    if hasattr(l, "merge_profile"):
                        assert len(p[6][p[0].index(l)]) == len(new_sub_profile[idx])
                        new_sub_profile[idx] = l.merge_profile(
                            new_sub_profile[idx], p[6][p[0].index(l)]
                        )
                    else:
                        new_sub_profile[idx] = None
                continue
            if l in prof1[0]:
                p = prof1
            else:
                p = prof2
            new_t.append(p[1][p[0].index(l)])
            idx = p[0].index(l)
            new_l.append(l)
            new_sub_profile.append(p[6][idx])

        new_rewrite = SequentialGraphRewriter(*new_l)
        new_nb_nodes = []
        for p1, p2 in zip(prof1[8], prof2[8]):
            new_nb_nodes.append((p1[0] + p2[0], p1[1] + p2[1]))
        new_nb_nodes.extend(prof1[8][len(new_nb_nodes) :])
        new_nb_nodes.extend(prof2[8][len(new_nb_nodes) :])

        new_callbacks_times = merge_dict(prof1[9], prof2[9])
        # We need to assert based on the name as we merge also based on
        # the name.
        assert {l.name for l in prof1[0]}.issubset({l.name for l in new_l})
        assert {l.name for l in prof2[0]}.issubset({l.name for l in new_l})
        assert len(new_t) == len(new_rewrite) == len(new_sub_profile)
        return (
            new_rewrite,
            new_t,
            prof1[2] + prof2[2],
            prof1[3] + prof2[3],
            -1,
            -1,
            new_sub_profile,
            [],
            new_nb_nodes,
            new_callbacks_times,
        )


class MergeFeature(Feature):
    """Keeps track of variables in a `FunctionGraph` that cannot be merged together.

    That way, the `MergeOptimizer` can remember the result of the last
    merge-pass on the `FunctionGraph`.

    """

    def on_attach(self, fgraph):
        if hasattr(fgraph, "merge_feature"):
            raise AlreadyThere()

        fgraph.merge_feature = self

        self.seen_atomics = set()
        self.atomic_sig = AssocList()
        self.atomic_sig_inv = AssocList()

        # For all Apply nodes
        # Set of distinct (not mergeable) nodes
        self.nodes_seen = set()
        # Ordered set of distinct (not mergeable) nodes without any input
        self.noinput_nodes = OrderedSet()

        # Each element of scheduled is a list of list of (out, new_out) pairs.
        # Each list of pairs represent the substitution needed to replace all
        # the outputs of a node with the outputs of a replacement candidate.
        # Each node can have several candidates. For instance, if "node" has
        # 2 outputs, and there are 3 replacement candidates, we will have:
        # shelf.scheduled = [
        #    [[(node.out1, cand1.out1), (node.out2, cand1.out2)],
        #     [(node.out1, cand2.out1), (node.out2, cand2.out2)],
        #     [(node.out1, cand3.out1), (node.out2, cand3.out2)]]]
        self.scheduled = []

        # List of (node, candidate) pairs, where we tried to replace node by
        # candidate, but it failed. This is used to avoid infinite loops
        # during the replacement phase.
        self.blacklist = []

        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def clone(self):
        return type(self)()

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if node in self.nodes_seen:
            # If inputs to a node change, it's not guaranteed that the node is
            # distinct from the other nodes in `self.nodes_seen`.
            self.nodes_seen.discard(node)
            self.process_node(fgraph, node)

        if isinstance(new_r, AtomicVariable):
            self.process_atomic(fgraph, new_r)

    def on_import(self, fgraph, node, reason):
        for c in node.inputs:
            if isinstance(c, AtomicVariable):
                self.process_atomic(fgraph, c)

        self.process_node(fgraph, node)

    def on_prune(self, fgraph, node, reason):
        self.nodes_seen.discard(node)
        if not node.inputs:
            self.noinput_nodes.discard(node)
        for c in node.inputs:
            if isinstance(c, AtomicVariable) and len(fgraph.clients[c]) <= 1:
                # This was the last node using this constant
                sig = self.atomic_sig[c]
                self.atomic_sig.discard(c)
                self.atomic_sig_inv.discard(sig)
                self.seen_atomics.discard(id(c))

    def process_atomic(self, fgraph, c):
        """Check if an atomic `c` can be merged, and queue that replacement."""
        if id(c) in self.seen_atomics:
            return
        sig = c.merge_signature()
        other_c = self.atomic_sig_inv.get(sig, None)
        if other_c is not None:
            # multiple names will clobber each other..
            # we adopt convention to keep the last name
            if c.name:
                other_c.name = c.name
            self.scheduled.append([[(c, other_c, "merge")]])
        else:
            # this is a new constant
            self.atomic_sig[c] = sig
            self.atomic_sig_inv[sig] = c
            self.seen_atomics.add(id(c))

    def process_node(self, fgraph, node):
        r"""Check if a `node` can be merged, and queue that replacement.

        When `node` is changed we check for other nodes (via the clients map)
        that depend on the same inputs.  If any of those other nodes have the
        same inputs and `Op` as `node`, they are queued to be merged.

        """

        if node in self.nodes_seen:
            return

        if node.inputs:
            # We use the smallest clients list.  Some `Op`s like `Elemwise`
            # have rewrites that put constants as the first inputs.  Since
            # constants generally have more clients than other types of nodes,
            # using `node.inputs[0]` will make us look at more nodes on
            # average, so by picking the smallest clients list, we might speed
            # things up?

            clients = sorted(
                (fgraph.clients[inp] for inp in node.inputs), key=lambda x: len(x)
            )[0]
            assert len(clients) > 0

            merge_candidates = [c for c, i in clients if c in self.nodes_seen]
        else:
            # If two nodes have no input, but perform the same operation,
            # they are not always constant-folded, so we want to merge them.
            # In that case, the candidates are all the nodes without inputs.
            merge_candidates = self.noinput_nodes

        replacement_candidates = []
        for candidate in merge_candidates:

            if candidate is node:
                continue
            if len(node.inputs) != len(candidate.inputs):
                continue

            inputs_match = all(
                node_in is cand_in
                for node_in, cand_in in zip(node.inputs, candidate.inputs)
            )

            if inputs_match and node.op == candidate.op:
                if (node, candidate) in self.blacklist:
                    # They were already tried, and there was an error
                    continue

                # Schedule transfer of clients from node to candidate
                pairs = list(
                    zip(
                        node.outputs,
                        candidate.outputs,
                        ["merge"] * len(node.outputs),
                    )
                )

                replacement_candidates.append(pairs)

        if replacement_candidates:
            self.scheduled.append(replacement_candidates)
        else:
            self.nodes_seen.add(node)
            if not node.inputs:
                self.noinput_nodes.add(node)


class MergeOptimizer(GraphRewriter):
    r"""Merges parts of the graph that are identical and redundant.

    The basic principle is that if two `Apply`\s have `Op`\s that compare equal, and
    identical inputs, then they do not both need to be computed. The clients of
    one are transferred to the other and one of them is removed from the graph.
    This procedure is carried out in input-to-output order throughout the graph.

    The first step of merging is atomic variable-merging, so that all clients of a
    :class:`Constant` like ``int(1)``, are transferred to just one particular
    instance of ``int(1)``.  :class:`NominalVariable`\s are not merged individually
    like this; only the nodes that use them are.

    """

    def add_requirements(self, fgraph):
        if not hasattr(fgraph, "merge_feature"):
            fgraph.attach_feature(MergeFeature())

    def apply(self, fgraph):
        sched = fgraph.merge_feature.scheduled
        nb_fail = 0
        t0 = time.time()
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callback_before = fgraph.execute_callbacks_time
            callbacks_before = fgraph.execute_callbacks_times.copy()

        nb_merged = 0
        nb_atomic = 0
        while sched:
            pairs_list = sched.pop()
            success = True
            for pairs_ in pairs_list:
                # We must check again the equivalence, as the graph could've
                # changed. If so, doing the replacement can introduce a node
                # that depends on itself.  Doing the full check of such cycles
                # every time is very time consuming. I think this double check
                # is faster than doing the full cycle check. The full cycle
                # check is skipped by `Validator.validate` if the graph doesn't
                # contain destroyers.
                var, candidate_var, merge_mode = pairs_[0]
                if merge_mode == "new_node" and var in fgraph.variables:
                    pass
                elif (
                    var not in fgraph.variables or candidate_var not in fgraph.variables
                ):
                    continue

                # Keep len(item) == 2 for item in pairs
                pairs = [pair[:2] for pair in pairs_]

                if var.owner and candidate_var.owner:
                    if merge_mode == "new_node":
                        inputs_match = True
                    else:
                        inputs_match = all(
                            node_in is cand_in
                            for node_in, cand_in in zip(
                                var.owner.inputs, candidate_var.owner.inputs
                            )
                        )

                    # No need to compare the op again, as it don't change.
                    if not inputs_match:
                        continue

                    if hasattr(fgraph, "destroy_handler"):
                        # If both nodes have clients that destroy them, we
                        # can't merge them.
                        clients = (
                            fgraph.clients[pairs[0][0]] + fgraph.clients[pairs[0][1]]
                        )
                        if any(
                            i in flatten(c.op.destroy_map.values())
                            for c, i in clients
                            if c != "output" and c.op.destroy_map
                        ):
                            continue

                if len(pairs) == 1 and pairs[0][0].type != pairs[0][1].type:
                    res = pairs[0][0].type.convert_variable(pairs[0][1])

                    # Since the fgraph.replace only checks the convert_variable
                    # in one way, we change the order in the case that
                    # convert_variable will not be successful.
                    if not res:
                        pairs = [(pairs[0][1], pairs[0][0])]

                try:
                    # If they're all `AtomicVariable`s, there's no need to call validate.
                    if all(isinstance(old, AtomicVariable) for old, _ in pairs):
                        fgraph.replace_all(pairs, reason="MergeOptimizer")
                    else:
                        fgraph.replace_all_validate(pairs, reason="MergeOptimizer")
                except InconsistencyError:
                    success = False
                    nb_fail += 1
                    fgraph.merge_feature.blacklist.append(
                        (pairs[0][0].owner, pairs[0][1].owner)
                    )

                if success:
                    nb_merged += len(pairs)
                    if isinstance(pairs[0][0], AtomicVariable):
                        nb_atomic += 1
                    break

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    t = v - callbacks_before[k]
                    if t > 0:
                        callbacks_time[k] = t
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}

        fgraph.merge_feature.blacklist = []

        return (
            nb_fail,
            time.time() - t0,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_atomic,
        )

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def print_profile(cls, stream, prof, level=0):

        (
            nb_fail,
            replace_time,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_atomic,
        ) = prof

        validate_time = validate_time or float("nan")
        callback_time = callback_time or float("nan")

        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        print(
            blanc,
            f"  nb fail={nb_fail:5d} merged={nb_merged:5d} atomic={nb_atomic:5d}",
            file=stream,
        )
        print(
            blanc,
            f"  time replace={replace_time:2.2f} validate={validate_time:2.2f} callback={callback_time:2.2f}",
            file=stream,
        )
        if callback_time > 1:
            print(blanc, "  callbacks_time", file=stream)
            for i in sorted(callbacks_time.items(), key=lambda a: a[1]):
                if i[1] > 0:
                    # We want to have the __str__ called, so we can't
                    # just print i.
                    print(blanc, "      ", i[0], ",", i[1], file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        def merge_none_number(v1, v2):
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            return v1 + v2

        nb_fail = prof1[0] + prof2[0]
        replace_time = prof1[1] + prof2[1]
        validate_time = merge_none_number(prof1[2], prof2[2])
        callback_time = merge_none_number(prof1[3], prof2[3])
        callbacks_time = merge_dict(prof1[4], prof2[4])
        nb_merged = prof1[5] + prof2[5]
        nb_atomic = prof1[6] + prof2[6]
        return (
            nb_fail,
            replace_time,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_atomic,
        )


def pre_constant_merge(fgraph, variables):
    """Merge constants in the graphs given by `variables`.

    .. warning::

        This changes the nodes in a graph in-place!

    Parameters
    ----------
    fgraph
        A `FunctionGraph` instance in which some of these `variables` may
        reside.

        We want to avoid terms in `variables` that are contained in `fgraph`.
        The reason for that: it will break consistency of `fgraph` and its
        features (e.g. `ShapeFeature`).

    variables
        A list of nodes for which we want to merge constant inputs.

    Notes
    -----
    It is used to pre-merge nodes generated inside an rewrite.  It is
    useful if there are many such replacements to make, so that `DebugMode`
    will not check each of them.

    """
    seen_var = set()
    # signature -> variable (for constants)
    const_sig_inv = {}
    if isinstance(variables, Variable):
        variables = [variables]

    def recursive_merge(var):

        if var in seen_var:
            return var

        if not hasattr(var, "owner"):
            return var

        # We don't want to merge constants that are *within* the
        # `FunctionGraph`
        if var.owner in fgraph.apply_nodes:
            return var

        seen_var.add(var)

        if isinstance(var, Constant):
            sig = var.signature()

            if sig in const_sig_inv:
                return const_sig_inv[sig]

            const_sig_inv[sig] = var

            return var

        if var.owner:
            for idx, inp in enumerate(var.owner.inputs):
                # XXX: This is changing the graph in place!
                var.owner.inputs[idx] = recursive_merge(inp)
        return var

    return [recursive_merge(v) for v in variables]


class MetaNodeRewriter(NodeRewriter):
    r"""
    Base class for meta-rewriters that try a set of `NodeRewriter`\s
    to replace a node and choose the one that executes the fastest.

    If the error `MetaNodeRewriterSkip` is raised during
    compilation, we will skip that function compilation and not print
    the error.

    """

    def __init__(self):
        self.verbose = config.metaopt__verbose
        self.track_dict = defaultdict(lambda: [])
        self.tag_dict = defaultdict(lambda: [])
        self._tracks = []
        self.rewriters = []

    def register(self, rewriter: NodeRewriter, tag_list: IterableType[str]):
        self.rewriters.append(rewriter)

        tracks = rewriter.tracks()
        if tracks:
            for c in tracks:
                self.track_dict[c].append(rewriter)
                self._tracks.append(c)

        for tag in tag_list:
            self.tag_dict[tag].append(rewriter)

    def tracks(self):
        return self._tracks

    def transform(self, fgraph, node, *args, **kwargs):
        # safety check: depending on registration, tracks may have been ignored
        if self._tracks is not None:
            if not isinstance(node.op, tuple(self._tracks)):
                return
        # first, we need to provide dummy values for all inputs
        # to the node that are not shared variables anyway
        givens = {}
        missing = set()
        for input in node.inputs:
            if isinstance(input, aesara.compile.SharedVariable):
                pass
            elif hasattr(input.tag, "test_value"):
                givens[input] = aesara.shared(
                    input.type.filter(input.tag.test_value),
                    input.name,
                    shape=input.broadcastable,
                    borrow=True,
                )
            else:
                missing.add(input)
        if missing:
            givens.update(self.provide_inputs(node, missing))
            missing.difference_update(givens.keys())
        # ensure we have data for all input variables that need it
        if missing:
            if self.verbose > 0:
                print(
                    f"{self.__class__.__name__} cannot meta-rewrite {node}, "
                    f"{len(missing)} of {int(node.nin)} input shapes unknown"
                )
            return
        # now we can apply the different rewrites in turn,
        # compile the resulting subgraphs and time their execution
        if self.verbose > 1:
            print(
                f"{self.__class__.__name__} meta-rewriting {node} ({len(self.get_rewrites(node))} choices):"
            )
        timings = []
        for node_rewriter in self.get_rewrites(node):
            outputs = node_rewriter.transform(fgraph, node, *args, **kwargs)
            if outputs:
                try:
                    fn = aesara.function(
                        [], outputs, givens=givens, on_unused_input="ignore"
                    )
                    fn.trust_input = True
                    timing = min(self.time_call(fn) for _ in range(2))
                except MetaNodeRewriterSkip:
                    continue
                except Exception as e:
                    if self.verbose > 0:
                        print(f"* {node_rewriter}: exception", e)
                    continue
                else:
                    if self.verbose > 1:
                        print(f"* {node_rewriter}: {timing:.5g} sec")
                    timings.append((timing, outputs, node_rewriter))
            else:
                if self.verbose > 0:
                    print(f"* {node_rewriter}: not applicable")
        # finally, we choose the fastest one
        if timings:
            timings.sort()
            if self.verbose > 1:
                print(f"= {timings[0][2]}")
            return timings[0][1]
        return

    def provide_inputs(self, node, inputs):
        """Return a dictionary mapping some `inputs` to `SharedVariable` instances of with dummy values.

        The `node` argument can be inspected to infer required input shapes.

        """
        raise NotImplementedError()

    def get_rewrites(self, node):
        """Return the rewrites that apply to `node`.

        This uses ``self.track_dict[type(node.op)]`` by default.
        """
        return self.track_dict[type(node.op)]

    def time_call(self, fn):
        start = time.time()
        fn()
        return time.time() - start


class FromFunctionNodeRewriter(NodeRewriter):
    """A `NodeRewriter` constructed from a function."""

    def __init__(self, fn, tracks=None, requirements=()):
        self.fn = fn
        self._tracks = tracks
        self._tracked_types = (
            tuple(t for t in tracks if isinstance(t, type)) if tracks else ()
        )
        self.requirements = requirements

    def transform(self, fgraph, node):
        if self._tracks:
            if not (
                node.op in self._tracks or isinstance(node.op, self._tracked_types)
            ):
                return False

        return self.fn(fgraph, node)

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def tracks(self):
        return self._tracks

    def __str__(self):
        return getattr(self, "__name__", repr(self))

    def __repr__(self):
        return f"FromFunctionNodeRewriter({repr(self.fn)}, {repr(self._tracks)}, {repr(self.requirements)})"

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.transform} id={id(self)}", file=stream)


def node_rewriter(
    tracks: Optional[Sequence[Union[Op, type]]],
    inplace: bool = False,
    requirements: Optional[Tuple[type, ...]] = (),
):
    r"""A decorator used to construct `FromFunctionNodeRewriter` instances.

    Parameters
    ----------
    tracks
        The `Op` types or instances to which this rewrite applies.
        Use ``None`` instead of an empty list to have the rewrite apply to
        all `Op`\s.
    inplace
        A boolean indicating whether or not the rewrite works in-place.
        If ``True``, a `DestroyHandler` `Feature` is added automatically added
        to the `FunctionGraph`\s applied to this rewrite.
    requirements
        `Feature` types required by this rewrite.

    """

    if requirements is None:
        requirements = ()

    def decorator(f):
        if tracks is not None:
            if len(tracks) == 0:
                raise ValueError(
                    "Use `None` instead of an empty list to make an rewrite apply to all nodes."
                )
            for t in tracks:
                if not (
                    isinstance(t, Op) or (isinstance(t, type) and issubclass(t, Op))
                ):
                    raise TypeError(
                        "`tracks` must consist of `Op` classes or instances."
                    )
        req = requirements
        if inplace:
            dh_handler = dh.DestroyHandler
            req = tuple(requirements) + (
                lambda fgraph: fgraph.attach_feature(dh_handler()),
            )
        rval = FromFunctionNodeRewriter(f, tracks, req)
        rval.__name__ = f.__name__
        return rval

    return decorator


class OpToRewriterTracker:
    r"""A container that maps `NodeRewriter`\s to `Op` instances and `Op`-type inheritance."""

    def __init__(self):
        self.tracked_instances: Dict[Op, List[NodeRewriter]] = {}
        self.tracked_types: Dict[type, List[NodeRewriter]] = {}
        self.untracked_rewrites: List[NodeRewriter] = []

    def add_tracker(self, rw: NodeRewriter):
        """Add a `NodeRewriter` to be keyed by its `NodeRewriter.tracks` or applied generally."""
        tracks = rw.tracks()

        if tracks is None:
            self.untracked_rewrites.append(rw)
        else:
            for c in tracks:
                if isinstance(c, type):
                    self.tracked_types.setdefault(c, []).append(rw)
                else:
                    self.tracked_instances.setdefault(c, []).append(rw)

    def _find_impl(self, cls) -> List[NodeRewriter]:
        r"""Returns the `NodeRewriter`\s that apply to `cls` based on inheritance.

        This based on `functools._find_impl`.
        """
        mro = _compose_mro(cls, self.tracked_types.keys())
        matches = []
        for t in mro:
            match = self.tracked_types.get(t, None)
            if match:
                matches.extend(match)
        return matches

    @functools.lru_cache()
    def get_trackers(self, op: Op) -> List[NodeRewriter]:
        """Get all the rewrites applicable to `op`."""
        return (
            self._find_impl(type(op))
            + self.tracked_instances.get(op, [])
            + self.untracked_rewrites
        )

    def get_rewriters(self):
        return chain(
            chain.from_iterable(
                chain(self.tracked_types.values(), self.tracked_instances.values())
            ),
            self.untracked_rewrites,
        )


class SequentialNodeRewriter(NodeRewriter):
    r"""An rewriter that applies a list of `NodeRewriter`\s to a node.

    Attributes
    ----------
    reentrant : bool
        Some global rewriters, like `NodeProcessingGraphRewriter`, use this value to
        determine if they should ignore new nodes.
    retains_inputs : bool
        States whether or not the inputs of a transformed node are transferred
        to the outputs.
    """

    def __init__(
        self,
        *rewriters: Rewriter,
        apply_all_rewrites: bool = False,
        profile: bool = False,
    ):
        """

        Parameters
        ----------
        rewriters
            A list of rewriters to be applied to nodes.
        apply_all_rewrites
            If ``False``, it will return after the first successfully applied
            rewrite; otherwise, it will apply every applicable rewrite
            incrementally.
        profile
            Whether or not to profile the rewrites.

        """
        super().__init__()

        self.rewrites: Sequence[Rewriter] = rewriters
        assert isinstance(self.rewrites, tuple)

        self.reentrant = any(
            getattr(rewrite, "reentrant", True) for rewrite in rewriters
        )
        self.retains_inputs = all(
            getattr(rewrite, "retains_inputs", False) for rewrite in rewriters
        )

        self.apply_all_rewrites = apply_all_rewrites

        self.profile = profile
        if self.profile:
            self.time_rewrites: Dict[Rewriter, float] = {}
            self.process_count: Dict[Rewriter, int] = {}
            self.applied_true: Dict[Rewriter, int] = {}
            self.node_created: Dict[Rewriter, int] = {}

        self.tracker = OpToRewriterTracker()

        for o in self.rewrites:

            self.tracker.add_tracker(o)

            if self.profile:
                self.time_rewrites.setdefault(o, 0.0)
                self.process_count.setdefault(o, 0)
                self.applied_true.setdefault(o, 0)
                self.node_created.setdefault(o, 0)

    def __str__(self):
        return getattr(
            self,
            "__name__",
            f"{type(self).__name__}({','.join([str(o) for o in self.rewrites])})",
        )

    def tracks(self):
        t = []
        for l in self.rewrites:
            at = l.tracks()
            if at:
                t.extend(at)
        return t

    def transform(self, fgraph, node):
        if len(self.rewrites) == 0:
            return

        repl = None

        while True:
            rewrites = self.tracker.get_trackers(node.op)

            new_repl = None
            for rewrite in rewrites:
                rewrite_start = time.time()
                new_repl = rewrite.transform(fgraph, node)
                rewrite_finish = time.time()
                if self.profile:
                    self.time_rewrites[rewrite] += rewrite_start - rewrite_finish
                    self.process_count[rewrite] += 1
                if not new_repl:
                    continue
                if isinstance(new_repl, (tuple, list)):
                    new_vars = new_repl
                else:  # It must be a dict
                    new_vars = list(new_repl.values())

                if config.optimizer_verbose:
                    print(
                        f"rewriting: rewrite {rewrite} replaces node {node} with {new_repl}"
                    )

                if self.profile:
                    self.node_created[rewrite] += len(
                        list(applys_between(fgraph.variables, new_vars))
                    )
                    self.applied_true[rewrite] += 1
                break
            if not new_repl:  # No rewrites applied in the last iteration
                return repl
            # only 1 iteration
            if not self.apply_all_rewrites:
                return new_repl
            if not new_vars[0].owner:
                # We are at the start of the graph.
                return new_repl
            if len(new_repl) > 1:
                s = {v.owner for v in new_repl}
                assert len(s) == 1
            repl = new_repl
            node = new_vars[0].owner

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        (time_rewrites, process_count, applied_true, node_created, profile) = prof

        if not profile:
            return

        blanc = "    " * int(level)
        print(blanc, f"{cls.__name__}", file=stream)
        print(blanc, "---------------------", file=stream)
        count_rewrite = []
        not_used = []
        not_used_time = 0
        for o, count in process_count.items():
            if count > 0:
                count_rewrite.append(
                    (time_rewrites[o], applied_true[o], count, o, node_created[o])
                )
            else:
                not_used.append((time_rewrites[o], o))
                not_used_time += time_rewrites[o]
        if count_rewrite:
            print(
                blanc,
                "  time taken - times applied - times tried - name - node_created:",
                file=stream,
            )
            count_rewrite.sort()
            for (t, a_t, count, o, n_c) in count_rewrite[::-1]:
                print(
                    blanc,
                    f"  {t:.3f}s - {int(a_t)} - {int(count)} - {o} - {int(n_c)}",
                    file=stream,
                )
            print(
                blanc,
                (
                    f"  {not_used_time:.3f}s - in {len(not_used)} rewrite(s) that were not used "
                    "(displaying only those with a runtime greater than 0)"
                ),
                file=stream,
            )
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip rewrites that have 0 times; they probably weren't even tried.
                    print(blanc + "  ", f"  {t:.3f}s - {o}", file=stream)
        else:
            print(blanc, " The rewriter wasn't successful ", file=stream)

        print(file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        raise NotImplementedError

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)
        if depth != 0:
            depth -= 1
            for lrewrite in self.rewrites:
                lrewrite.print_summary(stream, level=(level + 2), depth=depth)

    def add_requirements(self, fgraph):
        for rewrite in self.rewrites:
            rewrite.add_requirements(fgraph)


class SubstitutionNodeRewriter(NodeRewriter):
    """

    Replaces the application of a certain `Op` by the application of
    another `Op` that takes the same inputs as what it is replacing.

    Parameters
    ----------
    op1, op2
        ``op1.make_node`` and ``op2.make_node`` must take the same number of
        inputs and have the same number of outputs.

    Examples
    --------

        SubstitutionNodeRewriter(add, sub) ==>
            add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))

    """

    # an SubstitutionNodeRewriter does not apply to the nodes it produces
    reentrant = False
    # all the inputs of the original node are transferred to the outputs
    retains_inputs = True

    def __init__(self, op1, op2, transfer_tags=True):
        self.op1 = op1
        self.op2 = op2
        self.transfer_tags = transfer_tags

    def op_key(self):
        return self.op1

    def tracks(self):
        return [self.op1]

    def transform(self, fgraph, node):
        if node.op != self.op1:
            return False
        repl = self.op2.make_node(*node.inputs)
        if self.transfer_tags:
            repl.tag = copy.copy(node.tag)
            for output, new_output in zip(node.outputs, repl.outputs):
                new_output.tag = copy.copy(output.tag)
        return repl.outputs

    def __str__(self):
        return f"{self.op1} -> {self.op2}"


class RemovalNodeRewriter(NodeRewriter):
    """
    Removes all applications of an `Op` by transferring each of its
    outputs to the corresponding input.

    """

    reentrant = False  # no nodes are added at all

    def __init__(self, op):
        self.op = op

    def op_key(self):
        return self.op

    def tracks(self):
        return [self.op]

    def transform(self, fgraph, node):
        if node.op != self.op:
            return False
        return node.inputs

    def __str__(self):
        return f"{self.op}(x) -> x"

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__}(self.op) id={id(self)}",
            file=stream,
        )


class PatternNodeRewriter(NodeRewriter):
    """Replace all occurrences of an input pattern with an output pattern.

    The input and output patterns have the following syntax:

        input_pattern ::= (op, <sub_pattern1>, <sub_pattern2>, ...)
        input_pattern ::= dict(pattern = <input_pattern>,
                               constraint = <constraint>)
        sub_pattern ::= input_pattern
        sub_pattern ::= string
        sub_pattern ::= a Constant instance
        sub_pattern ::= int
        sub_pattern ::= float
        constraint ::= lambda fgraph, expr: additional matching condition

        output_pattern ::= (op, <output_pattern1>, <output_pattern2>, ...)
        output_pattern ::= string
        output_pattern ::= int
        output_pattern ::= float

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant variable in the input pattern, there will be a
    match iff a constant variable with the same value and the same type
    is found in its place.

    You can add a constraint to the match by using the ``dict(...)`` form
    described above with a ``'constraint'`` key. The constraint must be a
    function that takes the fgraph and the current Variable that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    The constructor creates a `PatternNodeRewriter` that replaces occurrences of
    `in_pattern` by occurrences of `out_pattern`.

    Examples
    --------

        PatternNodeRewriter((add, 'x', 'y'), (add, 'y', 'x'))
        PatternNodeRewriter((multiply, 'x', 'x'), (square, 'x'))
        PatternNodeRewriter((subtract, (add, 'x', 'y'), 'y'), 'x')
        PatternNodeRewriter((power, 'x', Constant(double, 2.0)), (square, 'x'))
        PatternNodeRewriter((boggle, {'pattern': 'x',
                            'constraint': lambda expr: expr.type == scrabble}),
                   (scrabble, 'x'))

    """

    def __init__(
        self,
        in_pattern,
        out_pattern,
        allow_multiple_clients: bool = False,
        skip_identities_fn=None,
        name: Optional[str] = None,
        tracks=(),
        get_nodes=None,
        values_eq_approx=None,
    ):
        """

        Parameters
        ----------
        in_pattern
            The input pattern that we want to replace.
        out_pattern
            The replacement pattern.
        allow_multiple_clients
            If ``False``, the pattern matching will fail if one of the subpatterns has
            more than one client.
        skip_identities_fn
            TODO
        name
            Set the name of this rewriter.
        tracks
            The values that :meth:`self.tracks` will return.
        get_nodes
            If you provide `tracks`, you must provide this parameter. It must be a
            function that takes the tracked node and returns a list of nodes on
            which we will try this rewrite.

        Notes
        -----
        `tracks` and `get_nodes` can be used to make this rewrite track a less
        frequent `Op`, which will prevent the rewrite from being tried as
        often.

        """
        from aesara.graph.rewriting.unify import convert_strs_to_vars

        var_map: Dict[str, "Var"] = {}
        self.in_pattern = convert_strs_to_vars(in_pattern, var_map=var_map)
        self.out_pattern = convert_strs_to_vars(out_pattern, var_map=var_map)
        self.values_eq_approx = values_eq_approx
        if isinstance(in_pattern, (list, tuple)):
            self.op = self.in_pattern[0]
        elif isinstance(in_pattern, dict):
            self.op = self.in_pattern["pattern"][0]
        else:
            raise TypeError(
                "The pattern to search for must start with a specific Op instance."
            )
        self.__doc__ = f"{self.__class__.__doc__}\n\nThis instance does: {self}\n"
        self.allow_multiple_clients = allow_multiple_clients
        self.skip_identities_fn = skip_identities_fn
        if name:
            self.__name__ = name
        self._tracks = tracks
        self.get_nodes = get_nodes
        if tracks != ():
            assert get_nodes

    def op_key(self):
        return self.op

    def tracks(self):
        if self._tracks != ():
            return self._tracks
        return [self.op]

    def transform(self, fgraph, node, get_nodes=True):
        """Check if the graph from node corresponds to ``in_pattern``.

        If it does, it constructs ``out_pattern`` and performs the replacement.

        """
        from etuples.core import ExpressionTuple
        from unification import reify, unify

        # TODO: We shouldn't need to iterate like this.
        if not self.allow_multiple_clients and any(
            len(fgraph.clients.get(v)) > 1
            for v in vars_between(fgraph.inputs, node.outputs)
            if v not in fgraph.inputs
        ):
            return False

        if get_nodes and self.get_nodes is not None:
            for real_node in self.get_nodes(fgraph, node):
                if real_node == "output":
                    continue
                ret = self.transform(fgraph, real_node, get_nodes=False)
                if ret is not False and ret is not None:
                    return dict(zip(real_node.outputs, ret))

        if node.op != self.op:
            return False

        s = unify(self.in_pattern, node.out)

        if s is False:
            return False

        ret = reify(self.out_pattern, s)

        if isinstance(ret, ExpressionTuple):
            ret = ret.evaled_obj

        if self.values_eq_approx:
            ret.tag.values_eq_approx = self.values_eq_approx

        if ret.owner:
            if not (
                len(node.outputs) == len(ret.owner.outputs)
                and all(
                    o.type.is_super(new_o.type)
                    for o, new_o in zip(node.outputs, ret.owner.outputs)
                )
            ):
                return False
        else:
            # ret is just an input variable
            assert len(node.outputs) == 1
            if not node.outputs[0].type.is_super(ret.type):
                return False

        return [ret]

    def __str__(self):
        if getattr(self, "__name__", None):
            return self.__name__

        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "{}({})".format(
                    str(pattern[0]),
                    ", ".join([pattern_to_str(p) for p in pattern[1:]]),
                )
            elif isinstance(pattern, dict):
                return "{} subject to {}".format(
                    pattern_to_str(pattern["pattern"]),
                    str(pattern.get("constraint", "no conditions")),
                )
            else:
                return str(pattern)

        return "{} -> {}".format(
            pattern_to_str(self.in_pattern),
            pattern_to_str(self.out_pattern),
        )

    def __repr__(self):
        return str(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "__name__", getattr(self, "name", None))
        print(
            f"{' ' * level}{self.__class__.__name__} {name}({self.in_pattern}, {self.out_pattern}) id={id(self)}",
            file=stream,
        )


class DispatchingFeature(Feature):
    """A `Feature` consisting of user-defined functions implementing each `Feature` callback method."""

    def __init__(self, importer, pruner, chin, name=None):
        self.importer = importer
        self.pruner = pruner
        self.chin = chin
        self.name = name

    def __str__(self):
        return f"{type(self).__name__}{{{self.name}}}"

    def on_import(self, fgraph, node, reason):
        if self.importer:
            self.importer(node)

    def on_prune(self, fgraph, node, reason):
        if self.pruner:
            self.pruner(node)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if self.chin:
            self.chin(node, i, r, new_r, reason)

    def on_detach(self, fgraph):
        # To allow pickling this object
        self.importer = None
        self.pruner = None
        self.chin = None


class NodeProcessingGraphRewriter(GraphRewriter):
    r"""A class providing a base implementation for applying `NodeRewriter.transform` results to a graph.

    This rewriter accepts the output of `NodeRewriter.transform`
    implementations and applies them to a `FunctionGraph`.

    It accepts a sequence of new output nodes or ``dict``s.  Entries in
    these ``dict``\s can be `Variable`\s and their new values.  It also accepts
    a special ``"remove"`` key.  A sequence of `Variable`\s mapped to the key
    ``"remove"`` are removed from the `FunctionGraph`.

    It also adds some interface elements for simple reentrant/recursive
    application of rewrites.  The parameter `NodeRewriter.ignore_newtrees` is
    intended to be used by subclasses, alongside the
    `NodeRewriter.attach_updater` and `NodeRewriter.detach_updater` methods, to
    determine whether or not sub-graphs created by rewrites are to have the
    same rewrites applied to them.

    """

    @classmethod
    def warn(cls, exc, nav, repl_pairs, node_rewriter, node):
        """A failure callback that prints a traceback."""
        if config.on_opt_error != "ignore":
            _logger.error(f"Rewrite failure due to: {node_rewriter}")
            _logger.error(f"node: {node}")
            _logger.error("TRACEBACK:")
            _logger.error(traceback.format_exc())
        if config.on_opt_error == "pdb":
            pdb.post_mortem(sys.exc_info()[2])
        elif isinstance(exc, AssertionError) or config.on_opt_error == "raise":
            # We always crash on AssertionError because something may be
            # seriously wrong if such an exception is raised.
            raise exc

    @classmethod
    def warn_inplace(cls, exc, nav, repl_pairs, node_rewriter, node):
        r"""A failure callback that ignores `InconsistencyError`\s and prints a traceback.

        If the error occurred during replacement, `repl_pairs` is set;
        otherwise, its value is ``None``.

        """
        if isinstance(exc, InconsistencyError):
            return
        return cls.warn(exc, nav, repl_pairs, node_rewriter, node)

    @classmethod
    def warn_ignore(cls, exc, nav, repl_pairs, node_rewriter, node):
        """A failure callback that ignores all errors."""

    def __init__(
        self,
        node_rewriter: Optional[NodeRewriter],
        ignore_newtrees: Literal[True, False, "auto"],
        failure_callback: Optional[FailureCallbackType] = None,
    ):
        """

        Parameters
        ----------
        node_rewriter
            A `NodeRewriter` to apply over a `FunctionGraph` (or ``None``).
        ignore_newtrees
            - ``True``: new subgraphs returned by an `NodeRewriter` are not a
            candidate for rewriting.
            - ``False``: new subgraphs returned by an `NodeRewriter` is a
            candidate for rewriting.
            - ``'auto'``: let the `node_rewriter` set this parameter via its
            :attr:`reentrant` attribute.
        failure_callback
            A function with the signature
            ``(exception, navigator, [(old, new), (old,new),...])``
            that is called when there's an exception.

            If the exception is raised in `node_rewriter.transform`, the
            ``new`` variables will be ``None``.

            If the exception is raised during validation (e.g. the new types
            don't match) then the new variables will be the ones created by
            ``self.transform``.

            If this parameter is ``None``, then exceptions are not caught here
            and are raised normally.

        """
        self.node_rewriter = node_rewriter
        if ignore_newtrees == "auto":
            self.ignore_newtrees = not getattr(node_rewriter, "reentrant", True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback
        super().__init__()

    def attach_updater(
        self,
        fgraph: FunctionGraph,
        importer: Optional[Callable],
        pruner: Optional[Callable],
        chin: Optional[Callable] = None,
        name: Optional[str] = None,
    ) -> Optional[DispatchingFeature]:
        r"""Install `FunctionGraph` listeners to help the navigator deal with the recursion-related functionality.

        Parameters
        ----------
        importer
            Function to be called when a rewrite adds something to the graph.
        pruner
            Function to be called when a rewrite removes something from the
            graph.
        chin
            Function to be called when a node's inputs change.
        name
            Name of the `DispatchingFeature` to attach.

        Returns
        -------
        The `FunctionGraph` plugin that handles the three tasks.
        Keep this around so that `Feature`\s can be detached later.

        """
        if self.ignore_newtrees:
            importer = None

        if importer is None and pruner is None:
            return None

        u = DispatchingFeature(importer, pruner, chin, name=name)
        fgraph.attach_feature(u)
        return u

    def detach_updater(
        self, fgraph: FunctionGraph, updater: Optional[DispatchingFeature]
    ):
        """Undo the work of `attach_updater`.

        Parameters
        ----------
        fgraph
            The `FunctionGraph`.
        updater
            The `DispatchingFeature` to remove.

        Returns
        -------
        None

        """
        if updater is not None:
            fgraph.remove_feature(updater)

    def process_node(
        self,
        fgraph: FunctionGraph,
        node: Apply,
        node_rewriter: Optional[NodeRewriter] = None,
    ):
        r"""Apply `node_rewriter` to `node`.

        The :meth:`node_rewriter.transform` method will return either ``False``, a
        list of `Variable`\s that are intended to replace :attr:`node.outputs`, or
        a ``dict`` specifying replacements--or the key ``"remove"`` mapped to a
        sequence of `Variable`\s to be removed.

        Parameters
        ----------
        fgraph
            A `FunctionGraph`.
        node
            An `Apply` instance in `fgraph`
        node_rewriter
            A `NodeRewriter` instance that may have a better idea for
            how to compute node's outputs.

        Returns
        -------
        bool
            If `fgraph` accepts the replacement, then the rewrite is
            successful and this function returns ``True``.  If there are no
            replacement candidates, or the `fgraph` rejects the replacements,
            this function returns ``False``.


        """
        node_rewriter = node_rewriter or self.node_rewriter
        # TODO FIXME: This class's interface is broken
        assert node_rewriter is not None
        try:
            replacements = node_rewriter.transform(fgraph, node)
        except Exception as e:
            if self.failure_callback is not None:
                self.failure_callback(
                    e, self, [(x, None) for x in node.outputs], node_rewriter, node
                )
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False
        old_vars = node.outputs
        remove: List[Variable] = []
        if isinstance(replacements, dict):
            if "remove" in replacements:
                remove = list(cast(Sequence[Variable], replacements.pop("remove")))
            old_vars = list(cast(Sequence[Variable], replacements.keys()))
            replacements = list(cast(Sequence[Variable], replacements.values()))
        elif not isinstance(replacements, (tuple, list)):
            raise TypeError(
                f"Node rewriter {node_rewriter} gave wrong type of replacement. "
                f"Expected list or tuple; got {replacements}"
            )
        if len(old_vars) != len(replacements):
            raise ValueError(
                f"Node rewriter {node_rewriter} gave wrong number of replacements"
            )
        # None in the replacement mean that this variable isn't used
        # and we want to remove it
        for r, rnew in zip(old_vars, replacements):
            if rnew is None and len(fgraph.clients[r]) > 0:
                raise ValueError(
                    f"Node rewriter {node_rewriter} tried to remove a variable"
                    f" that is being used: {r}"
                )
        # If an output would be replaced by itself, no need to perform
        # the replacement
        repl_pairs = [
            (r, rnew)
            for r, rnew in zip(old_vars, replacements)
            if rnew is not r and rnew is not None
        ]

        if len(repl_pairs) == 0:
            return False
        try:
            fgraph.replace_all_validate_remove(  # type: ignore
                repl_pairs, reason=node_rewriter, remove=remove
            )
            return True
        except Exception as e:
            # This means the replacements were rejected by the fgraph.
            #
            # This is not supposed to happen.  The default failure_callback
            # will print a traceback as a warning.
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs, node_rewriter, node)
                return False
            else:
                raise

    def add_requirements(self, fgraph):
        super().add_requirements(fgraph)
        # Added by default
        # fgraph.attach_feature(ReplaceValidate())
        if self.node_rewriter:
            self.node_rewriter.add_requirements(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)
        if depth != 0:
            self.node_rewriter.print_summary(
                stream, level=(level + 2), depth=(depth - 1)
            )


class WalkingGraphRewriter(NodeProcessingGraphRewriter):
    """A rewriter that applies a single `NodeRewriter` to each node in topological order (or reverse)."""

    def __init__(
        self,
        node_rewriter: NodeRewriter,
        order: Literal["out_to_in", "in_to_out"] = "in_to_out",
        ignore_newtrees: bool = False,
        failure_callback: Optional[FailureCallbackType] = None,
    ):
        if order not in ("out_to_in", "in_to_out"):
            raise ValueError("order must be 'out_to_in' or 'in_to_out'")
        self.order = order
        super().__init__(node_rewriter, ignore_newtrees, failure_callback)

    def apply(self, fgraph, start_from=None):
        if start_from is None:
            start_from = fgraph.outputs
        callback_before = fgraph.execute_callbacks_time
        nb_nodes_start = len(fgraph.apply_nodes)
        t0 = time.time()
        q = deque(io_toposort(fgraph.inputs, start_from))
        io_t = time.time() - t0

        def importer(node):
            if node is not current_node:
                q.append(node)

        u = self.attach_updater(
            fgraph, importer, None, name=getattr(self, "name", None)
        )
        nb = 0
        try:
            t0 = time.time()
            while q:
                if self.order == "out_to_in":
                    node = q.pop()
                else:
                    node = q.popleft()
                if node not in fgraph.apply_nodes:
                    continue
                current_node = node
                nb += self.process_node(fgraph, node)
            loop_t = time.time() - t0
        finally:
            self.detach_updater(fgraph, u)

        callback_time = fgraph.execute_callbacks_time - callback_before
        nb_nodes_end = len(fgraph.apply_nodes)
        return (
            self,
            nb,
            nb_nodes_start,
            nb_nodes_end,
            io_t,
            loop_t,
            callback_time,
            self.node_rewriter,
        )

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        if prof is None:  # Happen as merge_profile() isn't implemented
            print(blanc, f"{cls.__name__} merge_profile not implemented", file=stream)
            return

        (
            rewrite,
            nb,
            nb_nodes_start,
            nb_nodes_end,
            io_t,
            loop_t,
            callback_time,
            node_rewriter,
        ) = prof

        print(
            blanc,
            f"{cls.__name__} ",
            getattr(rewrite, "name", getattr(rewrite, "__name__", "")),
            file=stream,
        )

        print(
            blanc,
            "  nb_node (start, end, changed)",
            (nb_nodes_start, nb_nodes_end, nb),
            file=stream,
        )
        print(blanc, "  init io_toposort", io_t, file=stream)
        print(blanc, "  loop time", loop_t, file=stream)
        print(blanc, "  callback_time", callback_time, file=stream)
        if isinstance(node_rewriter, SequentialNodeRewriter):
            if node_rewriter.profile:
                node_rewriter.print_profile(
                    stream,
                    (
                        node_rewriter.time_rewrites,
                        node_rewriter.process_count,
                        node_rewriter.applied_true,
                        node_rewriter.node_created,
                        node_rewriter.profile,
                    ),
                    level=level + 1,
                )

    def __str__(self):
        return getattr(self, "__name__", super().__str__())


def walking_rewriter(
    order,
    *node_rewriters,
    name=None,
    failure_callback=WalkingGraphRewriter.warn_inplace,
    **kwargs,
):
    r"""Apply `node_rewriters` from the input/output nodes to the output/input nodes of a graph.

    This constructs `WalkingGraphRewriter`\s, and uses a `SequentialNodeRewriter` when there's
    more than one entry in `node_rewriters`.
    """
    if len(node_rewriters) > 1:
        # Don't wrap it uselessly if there is only one rewrite.
        node_rewriters = SequentialNodeRewriter(*node_rewriters)
    else:
        (node_rewriters,) = node_rewriters
        if not name:
            name = node_rewriters.__name__
    ret = WalkingGraphRewriter(
        node_rewriters,
        order=order,
        failure_callback=failure_callback,
        **kwargs,
    )
    if name:
        ret.__name__ = name
    return ret


in2out = partial(walking_rewriter, "in_to_out")
out2in = partial(walking_rewriter, "out_to_in")


class OpKeyGraphRewriter(NodeProcessingGraphRewriter):
    r"""A rewriter that applies a `NodeRewriter` to specific `Op`\s.

    The `Op`\s are provided by a :meth:`NodeRewriter.op_key` method (either
    as a list of `Op`\s or a single `Op`), and discovered within a
    `FunctionGraph` using the `NodeFinder` `Feature`.

    This is similar to the `Op`-based tracking feature used by other rewriters.

    """

    def __init__(self, node_rewriter, ignore_newtrees=False, failure_callback=None):
        if not hasattr(node_rewriter, "op_key"):
            raise TypeError(f"{node_rewriter} must have an `op_key` method.")
        super().__init__(node_rewriter, ignore_newtrees, failure_callback)

    def apply(self, fgraph):
        op = self.node_rewriter.op_key()
        if isinstance(op, (list, tuple)):
            q = reduce(list.__iadd__, map(fgraph.get_nodes, op))
        else:
            q = list(fgraph.get_nodes(op))

        def importer(node):
            if node is not current_node:
                if node.op == op:
                    q.append(node)

        u = self.attach_updater(
            fgraph, importer, None, name=getattr(self, "name", None)
        )
        try:
            while q:
                node = q.pop()
                if node not in fgraph.apply_nodes:
                    continue
                current_node = node
                self.process_node(fgraph, node)
        finally:
            self.detach_updater(fgraph, u)

    def add_requirements(self, fgraph):
        super().add_requirements(fgraph)
        fgraph.attach_feature(NodeFinder())


class ChangeTracker(Feature):
    def __init__(self):
        self.changed = False
        self.nb_imported = 0

    def clone(self):
        return type(self)()

    def on_import(self, fgraph, node, reason):
        self.nb_imported += 1
        self.changed = True

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        self.changed = True

    def reset(self):
        self.changed = False

    def on_attach(self, fgraph):
        if hasattr(fgraph, "change_tracker"):
            raise AlreadyThere()
        fgraph.change_tracker = self

    def on_detach(self, fgraph):
        del fgraph.change_tracker


def merge_dict(d1, d2):
    r"""Merge two ``dict``\s by adding their values."""
    d = d1.copy()
    for k, v in d2.items():
        if k in d:
            d[k] += v
        else:
            d[k] = v
    return d


class EquilibriumGraphRewriter(NodeProcessingGraphRewriter):
    """A `Rewriter` that applies its rewrites until a fixed-point/equilibrium is reached."""

    def __init__(
        self,
        rewriters: Sequence[Rewriter],
        failure_callback: Optional[FailureCallbackType] = None,
        ignore_newtrees: bool = True,
        tracks_on_change_inputs: bool = False,
        max_use_ratio: Optional[float] = None,
        final_rewriters: Optional[Sequence[GraphRewriter]] = None,
        cleanup_rewriters: Optional[Sequence[GraphRewriter]] = None,
    ):
        """

        Parameters
        ----------
        rewriters
            Node or graph rewriters to apply until equilibrium.
            The global rewriter will be run at the start of each iteration before
            the node rewriter.
        failure_callback
            See :attr:`NodeProcessingGraphRewriter.failure_callback`.
        ignore_newtrees
            See :attr:`NodeProcessingGraphRewriter.ignore_newtrees`.
        tracks_on_change_inputs
            See :attr:`NodeProcessingGraphRewriter.tracks_on_change_inputs`.
        max_use_ratio
            Each rewriter can be applied at most ``(size_of_graph * max_use_ratio)``
            times.
        final_rewriters
            Rewriters that will be run after each iteration.
        cleanup_rewriters
            Rewriters applied after all graph rewriters, then when one
            `NodeRewriter` is applied, then after all final rewriters.
            They should not traverse the entire graph, since they are called
            very frequently.  The `MergeOptimizer` is one example of a rewriter
            that respects this.

        """
        super().__init__(
            None, ignore_newtrees=ignore_newtrees, failure_callback=failure_callback
        )
        self.global_rewriters: List[GraphRewriter] = []
        self.tracks_on_change_inputs = tracks_on_change_inputs

        self.node_tracker = OpToRewriterTracker()

        for rewriter in rewriters:
            if isinstance(rewriter, NodeRewriter):
                self.node_tracker.add_tracker(rewriter)
            else:
                assert isinstance(rewriter, GraphRewriter)
                self.global_rewriters.append(rewriter)

        if final_rewriters:
            self.final_rewriters = list(final_rewriters)
        else:
            self.final_rewriters = []

        if cleanup_rewriters:
            self.cleanup_rewriters = list(cleanup_rewriters)
        else:
            self.cleanup_rewriters = []

        self.max_use_ratio = max_use_ratio

    def get_node_rewriters(self):
        yield from self.node_tracker.get_rewriters()

    def get_local_optimizers(self):
        warnings.warn(
            "`get_local_optimizers` is deprecated; use `get_node_rewriters` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.get_node_rewriters()

    def add_requirements(self, fgraph):
        super().add_requirements(fgraph)
        for rewriter in self.get_node_rewriters():
            rewriter.add_requirements(fgraph)
        for rewriter in self.global_rewriters:
            rewriter.add_requirements(fgraph)
        for rewriter in self.final_rewriters:
            rewriter.add_requirements(fgraph)
        for rewriter in self.cleanup_rewriters:
            rewriter.add_requirements(fgraph)

    def apply(self, fgraph, start_from=None):
        change_tracker = ChangeTracker()
        fgraph.attach_feature(change_tracker)
        if start_from is None:
            start_from = fgraph.outputs
        else:
            for node in start_from:
                assert node in fgraph.outputs

        changed = True
        max_use_abort = False
        rewriter_name = None
        global_process_count = {}
        start_nb_nodes = len(fgraph.apply_nodes)
        max_nb_nodes = len(fgraph.apply_nodes)
        max_use = max_nb_nodes * self.max_use_ratio

        loop_timing = []
        loop_process_count = []
        global_rewriter_timing = []
        time_rewriters = {}
        io_toposort_timing = []
        nb_nodes = []
        node_created = {}
        global_sub_profs = []
        final_sub_profs = []
        cleanup_sub_profs = []
        for rewriter in (
            self.global_rewriters
            + list(self.get_node_rewriters())
            + self.final_rewriters
            + self.cleanup_rewriters
        ):
            global_process_count.setdefault(rewriter, 0)
            time_rewriters.setdefault(rewriter, 0)
            node_created.setdefault(rewriter, 0)

        def apply_cleanup(profs_dict):
            changed = False
            for crewriter in self.cleanup_rewriters:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_rewrite = time.time()
                sub_prof = crewriter.apply(fgraph)
                time_rewriters[crewriter] += time.time() - t_rewrite
                profs_dict[crewriter].append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(crewriter, 0)
                    process_count[crewriter] += 1
                    global_process_count[crewriter] += 1
                    changed = True
                    node_created[crewriter] += change_tracker.nb_imported - nb
            return changed

        while changed and not max_use_abort:
            process_count = {}
            t0 = time.time()
            changed = False
            iter_cleanup_sub_profs = {}
            for crewrite in self.cleanup_rewriters:
                iter_cleanup_sub_profs[crewrite] = []

            # Apply global rewriters
            sub_profs = []
            for grewrite in self.global_rewriters:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_rewrite = time.time()
                sub_prof = grewrite.apply(fgraph)
                time_rewriters[grewrite] += time.time() - t_rewrite
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(grewrite, 0)
                    process_count[grewrite] += 1
                    global_process_count[grewrite] += 1
                    changed = True
                    node_created[grewrite] += change_tracker.nb_imported - nb
                    if global_process_count[grewrite] > max_use:
                        max_use_abort = True
                        rewriter_name = getattr(grewrite, "name", None) or getattr(
                            grewrite, "__name__", ""
                        )
            global_sub_profs.append(sub_profs)

            global_rewriter_timing.append(float(time.time() - t0))

            changed |= apply_cleanup(iter_cleanup_sub_profs)

            topo_t0 = time.time()
            q = deque(io_toposort(fgraph.inputs, start_from))
            io_toposort_timing.append(time.time() - topo_t0)

            nb_nodes.append(len(q))
            max_nb_nodes = max(max_nb_nodes, len(q))
            max_use = max_nb_nodes * self.max_use_ratio

            def importer(node):
                if node is not current_node:
                    q.append(node)

            chin = None
            if self.tracks_on_change_inputs:

                def chin(node, i, r, new_r, reason):
                    if node is not current_node and not isinstance(node, str):
                        q.append(node)

            u = self.attach_updater(
                fgraph, importer, None, chin=chin, name=getattr(self, "name", None)
            )
            try:
                while q:
                    node = q.pop()
                    if node not in fgraph.apply_nodes:
                        continue
                    current_node = node
                    for node_rewriter in self.node_tracker.get_trackers(node.op):
                        nb = change_tracker.nb_imported
                        t_rewrite = time.time()
                        node_rewriter_change = self.process_node(
                            fgraph, node, node_rewriter
                        )
                        time_rewriters[node_rewriter] += time.time() - t_rewrite
                        if not node_rewriter_change:
                            continue
                        process_count.setdefault(node_rewriter, 0)
                        process_count[node_rewriter] += 1
                        global_process_count[node_rewriter] += 1
                        changed = True
                        node_created[node_rewriter] += change_tracker.nb_imported - nb
                        changed |= apply_cleanup(iter_cleanup_sub_profs)
                        if global_process_count[node_rewriter] > max_use:
                            max_use_abort = True
                            rewriter_name = getattr(
                                node_rewriter, "name", None
                            ) or getattr(node_rewriter, "__name__", "")
                        if node not in fgraph.apply_nodes:
                            # go to next node
                            break
            finally:
                self.detach_updater(fgraph, u)

            # Apply final rewriters
            sub_profs = []
            t_before_final_rewrites = time.time()
            for grewrite in self.final_rewriters:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_rewrite = time.time()
                sub_prof = grewrite.apply(fgraph)
                time_rewriters[grewrite] += time.time() - t_rewrite
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(grewrite, 0)
                    process_count[grewrite] += 1
                    global_process_count[grewrite] += 1
                    changed = True
                    node_created[grewrite] += change_tracker.nb_imported - nb
                    if global_process_count[grewrite] > max_use:
                        max_use_abort = True
                        rewriter_name = getattr(grewrite, "name", None) or getattr(
                            grewrite, "__name__", ""
                        )
            final_sub_profs.append(sub_profs)

            global_rewriter_timing[-1] += time.time() - t_before_final_rewrites

            changed |= apply_cleanup(iter_cleanup_sub_profs)

            # Merge clean up profiles during that iteration
            c_sub_profs = []
            for crewrite, sub_profs in iter_cleanup_sub_profs.items():
                sub_prof = sub_profs[0]
                for s_p in sub_profs[1:]:
                    sub_prof = crewrite.merge_profile(sub_prof, s_p)
                c_sub_profs.append(sub_prof)
            cleanup_sub_profs.append(c_sub_profs)

            loop_process_count.append(process_count)
            loop_timing.append(float(time.time() - t0))

        end_nb_nodes = len(fgraph.apply_nodes)

        if max_use_abort:
            msg = (
                f"{type(self).__name__} max'ed out by {rewriter_name}."
                "You can safely raise the current threshold of "
                f"{config.optdb__max_use_ratio} with the option `optdb__max_use_ratio`."
            )
            if config.on_opt_error == "raise":
                raise AssertionError(msg)
            else:
                _logger.error(msg)
        fgraph.remove_feature(change_tracker)
        assert len(loop_process_count) == len(loop_timing)
        assert len(loop_process_count) == len(global_rewriter_timing)
        assert len(loop_process_count) == len(nb_nodes)
        assert len(loop_process_count) == len(io_toposort_timing)
        assert len(loop_process_count) == len(global_sub_profs)
        assert len(loop_process_count) == len(final_sub_profs)
        assert len(loop_process_count) == len(cleanup_sub_profs)
        return (
            self,
            loop_timing,
            loop_process_count,
            (start_nb_nodes, end_nb_nodes, max_nb_nodes),
            global_rewriter_timing,
            nb_nodes,
            time_rewriters,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        )

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}", file=stream
        )
        if depth != 0:
            for node_rewriter in self.get_node_rewriters():
                node_rewriter.print_summary(
                    stream, level=(level + 2), depth=(depth - 1)
                )

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        (
            rewrite,
            loop_timing,
            loop_process_count,
            (start_nb_nodes, end_nb_nodes, max_nb_nodes),
            global_rewrite_timing,
            nb_nodes,
            time_rewrites,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        ) = prof

        blanc = "    " * level
        print(blanc, cls.__name__, end=" ", file=stream)
        print(
            blanc,
            getattr(rewrite, "name", getattr(rewrite, "__name__", "")),
            file=stream,
        )
        print(
            blanc,
            f"  time {sum(loop_timing):.3f}s for {len(loop_timing)} passes",
            file=stream,
        )
        print(
            blanc,
            f"  nb nodes (start, end,  max) {int(start_nb_nodes)} {int(end_nb_nodes)} {int(max_nb_nodes)}",
            file=stream,
        )
        print(blanc, f"  time io_toposort {sum(io_toposort_timing):.3f}s", file=stream)
        s = sum(time_rewrites[o] for o in rewrite.get_node_rewriters())
        print(blanc, f"  time in node rewriters {s:.3f}s", file=stream)
        s = sum(time_rewrites[o] for o in rewrite.global_rewriters)
        print(blanc, f"  time in graph rewriters {s:.3f}s", file=stream)
        s = sum(time_rewrites[o] for o in rewrite.final_rewriters)
        print(blanc, f"  time in final rewriters {s:.3f}s", file=stream)
        s = sum(time_rewrites[o] for o in rewrite.cleanup_rewriters)
        print(blanc, f"  time in cleanup rewriters {s:.3f}s", file=stream)
        for i in range(len(loop_timing)):
            loop_times = ""
            if loop_process_count[i]:
                d = list(
                    reversed(sorted(loop_process_count[i].items(), key=lambda a: a[1]))
                )
                loop_times = " ".join([str((str(k), v)) for k, v in d[:5]])
                if len(d) > 5:
                    loop_times += " ..."
            print(
                blanc,
                (
                    f"  {int(i):2d} - {loop_timing[i]:.3f}s {int(sum(loop_process_count[i].values()))} ({global_rewrite_timing[i]:.3f}s in graph rewriters, "
                    f"{io_toposort_timing[i]:.3f}s io_toposort) - {int(nb_nodes[i])} nodes - {loop_times}"
                ),
                file=stream,
            )

        count_rewrite = []
        not_used = []
        not_used_time = 0
        process_count = {}
        for o in (
            rewrite.global_rewriters
            + list(rewrite.get_node_rewriters())
            + list(rewrite.final_rewriters)
            + list(rewrite.cleanup_rewriters)
        ):
            process_count.setdefault(o, 0)
        for count in loop_process_count:
            for o, v in count.items():
                process_count[o] += v
        for o, count in process_count.items():
            if count > 0:
                count_rewrite.append((time_rewrites[o], count, node_created[o], o))
            else:
                not_used.append((time_rewrites[o], o))
                not_used_time += time_rewrites[o]

        if count_rewrite:
            print(
                blanc, "  times - times applied - nb node created - name:", file=stream
            )
            count_rewrite.sort()
            for (t, count, n_created, o) in count_rewrite[::-1]:
                print(
                    blanc,
                    f"  {t:.3f}s - {int(count)} - {int(n_created)} - {o}",
                    file=stream,
                )
            print(
                blanc,
                f"  {not_used_time:.3f}s - in {len(not_used)} rewrites that were not used (i.e. those with a run-time of zero)",
                file=stream,
            )
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip rewrites that have no run-times; they probably weren't even tried.
                    print(blanc + "  ", f"  {t:.3f}s - {o}", file=stream)
            print(file=stream)
        gf_rewrites = [
            o
            for o in (
                rewrite.global_rewrites
                + list(rewrite.final_rewriters)
                + list(rewrite.cleanup_rewriters)
            )
            if o.print_profile.__code__ is not GraphRewriter.print_profile.__code__
        ]
        if not gf_rewrites:
            return
        print(blanc, "Global, final, and clean up rewriters", file=stream)
        for i in range(len(loop_timing)):
            print(blanc, f"Iter {int(i)}", file=stream)
            for o, prof in zip(rewrite.global_rewriters, global_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(rewrite.final_rewriters, final_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(rewrite.cleanup_rewriters, cleanup_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)

    @staticmethod
    def merge_profile(prof1, prof2):
        node_rewriters = OrderedSet(prof1[0].get_node_rewriters()).union(
            prof2[0].get_node_rewriters()
        )
        global_rewriters = OrderedSet(prof1[0].global_rewriters).union(
            prof2[0].global_rewriters
        )
        final_rewriters = list(
            OrderedSet(prof1[0].final_rewriters).union(prof2[0].final_rewriters)
        )
        cleanup_rewriters = list(
            OrderedSet(prof1[0].cleanup_rewriters).union(prof2[0].cleanup_rewriters)
        )
        new_rewriter = EquilibriumGraphRewriter(
            node_rewriters.union(global_rewriters),
            max_use_ratio=1,
            final_rewriters=final_rewriters,
            cleanup_rewriters=cleanup_rewriters,
        )

        def add_append_list(l1, l2):
            l = copy.copy(l1)
            for idx, nb in enumerate(l2):
                if idx < len(l):
                    l[idx] += nb
                else:
                    l.append(nb)
            return l

        loop_timing = add_append_list(prof1[1], prof2[1])

        loop_process_count = list(prof1[2])
        global_sub_profs = []
        final_sub_profs = []
        cleanup_sub_profs = []

        for i in range(min(len(loop_process_count), len(prof2[2]))):
            process_count = loop_process_count[i]
            for process, count in prof2[2][i].items():
                if process in process_count:
                    process_count[process] += count
                else:
                    process_count[process] = count

            def merge(rewriters, attr, idx):
                tmp = []
                for rewriter in rewriters:
                    o1 = getattr(prof1[0], attr)
                    o2 = getattr(prof2[0], attr)
                    if rewriter in o1 and rewriter in o2:
                        p1 = prof1[idx][i][o1.index(rewriter)]
                        p2 = prof2[idx][i][o2.index(rewriter)]
                        m = None
                        if hasattr(rewriter, "merge_profile"):
                            m = rewriter.merge_profile(p1, p2)
                    elif rewriter in o1:
                        m = prof1[idx][i][o1.index(rewriter)]
                    else:
                        m = prof2[idx][i][o2.index(rewriter)]
                    tmp.append(m)
                return tmp

            global_sub_profs.append(merge(global_rewriters, "global_rewriters", 9))
            final_sub_profs.append(merge(final_rewriters, "final_rewriters", 10))
            cleanup_sub_profs.append(merge(cleanup_rewriters, "cleanup_rewriters", 11))

        # Add the iteration done by only one of the profile.
        loop_process_count.extend(prof1[2][len(loop_process_count) :])
        global_sub_profs.extend(prof1[9][len(global_sub_profs) :])
        final_sub_profs.extend(prof1[10][len(final_sub_profs) :])
        cleanup_sub_profs.extend(prof1[11][len(cleanup_sub_profs) :])

        global_sub_profs.extend(prof2[9][len(loop_process_count) :])
        final_sub_profs.extend(prof2[10][len(loop_process_count) :])
        cleanup_sub_profs.extend(prof2[11][len(loop_process_count) :])

        max_nb_nodes = max(prof1[3], prof2[3])

        global_rewrite_timing = add_append_list(prof1[4], prof2[4])

        nb_nodes = add_append_list(prof1[5], prof2[5])

        time_rewrites = merge_dict(prof1[6], prof2[6])
        io_toposort_timing = add_append_list(prof1[7], prof2[7])
        assert (
            len(loop_timing)
            == len(global_rewrite_timing)
            == len(global_sub_profs)
            == len(io_toposort_timing)
            == len(nb_nodes)
        )
        assert len(loop_timing) == max(len(prof1[1]), len(prof2[1]))

        node_created = merge_dict(prof1[8], prof2[8])
        return (
            new_rewriter,
            loop_timing,
            loop_process_count,
            max_nb_nodes,
            global_rewrite_timing,
            nb_nodes,
            time_rewrites,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        )


def _check_chain(r, chain):
    """
    WRITEME

    """
    chain = list(reversed(chain))
    while chain:
        elem = chain.pop()
        if elem is None:
            if r.owner is not None:
                return False
        elif r.owner is None:
            return False
        elif isinstance(elem, Op):
            if r.owner.op != elem:
                return False
        else:
            try:
                if issubclass(elem, Op) and not isinstance(r.owner.op, elem):
                    return False
            except TypeError:
                return False
        if chain:
            r = r.owner.inputs[chain.pop()]
    # print 'check_chain', _check_chain.n_calls
    # _check_chain.n_calls += 1

    # The return value will be used as a Boolean, but some Variables cannot
    # be used as Booleans (the results of comparisons, for instance)
    return r is not None


def check_chain(r, *chain):
    """
    WRITEME

    """
    if isinstance(r, Apply):
        r = r.outputs[0]
    return _check_chain(r, reduce(list.__iadd__, ([x, 0] for x in chain)))


def pre_greedy_node_rewriter(
    fgraph: FunctionGraph, rewrites: Sequence[NodeRewriter], out: Variable
) -> Variable:
    """Apply node rewriters throughout a graph in a greedy, pre-traversal way.

    This function traverses the computation graph in the graph before the
    variable `out` but that are not in the `fgraph`. It applies
    `rewrites` to each variable on the traversed graph.

    .. warning::

        This changes the nodes in a graph in-place.

    Its main use is to apply locally constant folding when generating
    the graph of the indices of a `Subtensor`.

    Changes should not be applied to nodes that are in an `fgraph`,
    so we use `fgraph` to prevent that.

    Notes
    -----
    This doesn't do an equilibrium rewrite, so, if there is a rewrite--like
    `local_upcast_elemwise_constant_inputs`--in the list that adds additional
    nodes to the inputs of the node, it might be necessary to call this
    function multiple times.

    Parameters
    ----------
    fgraph
        The graph used to avoid/filter nodes.
    rewrites
        A sequence of rewrites to apply.
    out
        The graph to rewrite.

    """

    def local_recursive_function(
        rewrite_list: Sequence[NodeRewriter],
        out: Variable,
        rewritten_vars: Dict[Variable, Variable],
        depth: int,
    ) -> Tuple[List[Variable], Dict[Variable, Variable]]:
        if not getattr(out, "owner", None):
            return [out], rewritten_vars
        node = out.owner

        if node in fgraph.apply_nodes:
            return node.outputs, rewritten_vars

        # Walk up the graph via the node's inputs
        for idx, inp in enumerate(node.inputs):
            if inp in rewritten_vars:
                nw_in = rewritten_vars[inp]
            else:
                if inp.owner:
                    outs, rewritten_vars = local_recursive_function(
                        rewrite_list, inp, rewritten_vars, depth + 1
                    )
                    for k, v in zip(inp.owner.outputs, outs):
                        rewritten_vars[k] = v
                    nw_in = outs[inp.owner.outputs.index(inp)]

                else:
                    nw_in = inp
                    rewritten_vars[inp] = inp

            # XXX: An in-place change
            node.inputs[idx] = nw_in

        # Apply the rewrites
        results = node.outputs
        for rewrite in rewrite_list:
            ret = rewrite.transform(fgraph, node)
            if ret is not False and ret is not None:
                assert isinstance(ret, Sequence)
                assert len(ret) == len(node.outputs), rewrite
                for k, v in zip(node.outputs, ret):
                    rewritten_vars[k] = v
                results = ret
                if ret[0].owner:
                    node = out.owner
                else:
                    break

        return results, rewritten_vars

    if out.owner:
        out_index: int = out.owner.outputs.index(out)
    else:
        out_index = 0

    final_outs, rewritten_nodes = local_recursive_function(rewrites, out, {}, 0)
    return final_outs[out_index]


def copy_stack_trace(from_var, to_var):
    r"""Copy the stack traces from `from_var` to `to_var`.

    Parameters
    ----------
    from_var :
        `Variable` or list `Variable`\s to copy stack traces from.
    to_var :
        `Variable` or list `Variable`\s to copy stack traces to.

    Notes
    -----
    The stacktrace is assumed to be of the form of a list of lists
    of tuples. Each tuple contains the filename, line number, function name
    and so on. Each list of tuples contains the truples belonging to a
    particular `Variable`.

    """

    # Store stack traces from from_var
    tr = []
    if isinstance(from_var, Iterable) and not isinstance(from_var, Variable):
        # If from_var is a list, store concatenated stack traces
        for v in from_var:
            tr += getattr(v.tag, "trace", [])

    else:
        # If from_var is not a list, it must be a single tensor variable,
        # so just store that particular stack trace
        tr = getattr(from_var.tag, "trace", [])

    if tr and isinstance(tr[0], tuple):
        # There was one single stack trace, we encapsulate it in a list
        tr = [tr]

    # Copy over stack traces to to_var
    if isinstance(to_var, Iterable) and not isinstance(to_var, Variable):
        # Copy over stack traces from from_var to each variable in
        # to_var, including the stack_trace of the to_var before
        for v in to_var:
            v.tag.trace = getattr(v.tag, "trace", []) + tr
    else:
        # Copy over stack traces from from_var to each variable to
        # to_var, including the stack_trace of the to_var before
        to_var.tag.trace = getattr(to_var.tag, "trace", []) + tr
    return to_var


def check_stack_trace(f_or_fgraph, ops_to_check="last", bug_print="raise"):
    r"""Checks if the outputs of specific `Op`\s have a stack trace.

    Parameters
    ----------
    f_or_fgraph : Function or FunctionGraph
        The compiled function or the function graph to be analysed.
    ops_to_check
        This value can be of four different types:
            - classes or instances inheriting from `Op`
            - tuple/list of classes or instances inheriting from `Op`
            - string
            - function returning a boolean and taking as input an instance of `Op`

        - if `ops_to_check` is a string, it should be either ``'last'`` or ``'all'``.
          ``'last'`` will check only the last `Op` of the graph while ``'all'`` will
          check all the `Op`\s of the graph.
        - if `ops_to_check` is an `Op` or a tuple/list of `Op`\s, the function will
          check that all the outputs of their occurrences in the graph have a
          stack trace.
        - if `ops_to_check` is a function, it should take as input a
          `Op` and return a boolean indicating if the input `Op` should
          be checked or not.

    bug_print
        This value is a string belonging to ``{'raise', 'warn', 'ignore'}``.
        You can specify the behaviour of the function when the specified
        `ops_to_check` are not in the graph of `f_or_fgraph`: it can either raise
        an exception, write a warning or simply ignore it.

    Returns
    -------
    boolean
        ``True`` if the outputs of the specified ops have a stack, ``False``
        otherwise.

    """
    if isinstance(f_or_fgraph, aesara.compile.function.types.Function):
        fgraph = f_or_fgraph.maker.fgraph
    elif isinstance(f_or_fgraph, aesara.graph.fg.FunctionGraph):
        fgraph = f_or_fgraph
    else:
        raise ValueError("The type of f_or_fgraph is not supported")

    if isinstance(ops_to_check, Op) or (
        inspect.isclass(ops_to_check) and issubclass(ops_to_check, Op)
    ):
        ops_to_check = (ops_to_check,)

    # if ops_to_check is a string
    if isinstance(ops_to_check, str):
        if ops_to_check == "last":
            apply_nodes_to_check = [
                fgraph.outputs[i].owner for i in range(len(fgraph.outputs))
            ]
        elif ops_to_check == "all":
            apply_nodes_to_check = fgraph.apply_nodes
        else:
            raise ValueError("The string ops_to_check is not recognised")

    # if ops_to_check is a list/tuple of ops
    elif isinstance(ops_to_check, (tuple, list)):
        # Separate classes from instances in ops_to_check
        op_instances = []
        op_classes = []
        for obj in ops_to_check:
            if isinstance(obj, Op):
                op_instances.append(obj)
            else:
                op_classes.append(obj)
        op_classes = tuple(op_classes)

        apply_nodes_to_check = [
            node for node in fgraph.apply_nodes if node.op in ops_to_check
        ] + [
            node
            for node in fgraph.apply_nodes
            if isinstance(node.op, op_classes)
            or (
                hasattr(node.op, "scalar_op")
                and isinstance(node.op.scalar_op, op_classes)
            )
        ]

    # if ops_to_check is a function
    elif callable(ops_to_check):
        apply_nodes_to_check = [
            node for node in fgraph.apply_nodes if ops_to_check(node)
        ]

    else:
        raise ValueError("ops_to_check does not have the right type")

    if not apply_nodes_to_check:
        msg = (
            "Provided op instances/classes are not in the graph or the "
            "graph is empty"
        )
        if bug_print == "warn":
            warnings.warn(msg)
        elif bug_print == "raise":
            raise Exception(msg)
        elif bug_print == "ignore":
            pass
        else:
            raise ValueError("The string bug_print is not recognised")

    for node in apply_nodes_to_check:
        for output in node.outputs:
            if not hasattr(output.tag, "trace") or not output.tag.trace:
                return False

    return True


class CheckStackTraceFeature(Feature):
    def on_import(self, fgraph, node, reason):
        # In `optdb` we only register the `CheckStackTraceRewriter` when
        # `config.check_stack_trace` is not off, but we also double check here.
        if config.check_stack_trace != "off" and not check_stack_trace(fgraph, "all"):
            if config.check_stack_trace == "raise":
                raise AssertionError(
                    f"Empty stack trace. The rewrite that inserted this variable is {reason}."
                )
            elif config.check_stack_trace in ("log", "warn"):
                apply_nodes_to_check = fgraph.apply_nodes
                for node in apply_nodes_to_check:
                    for output in node.outputs:
                        if not hasattr(output.tag, "trace") or not output.tag.trace:
                            output.tag.trace = [
                                [
                                    (
                                        "",
                                        0,
                                        f"Empty stack trace. The rewrite that inserted this variable is {reason}.",
                                        "",
                                    )
                                ]
                            ]
                if config.check_stack_trace == "warn":
                    warnings.warn(
                        f"Empty stack trace. The rewrite that inserted this variable is {reason}."
                    )


class CheckStackTraceRewriter(GraphRewriter):
    """Rewriter that serves to add `CheckStackTraceRewriter` as a feature."""

    def add_requirements(self, fgraph):
        if not hasattr(fgraph, "CheckStackTraceFeature"):
            fgraph.attach_feature(CheckStackTraceFeature())

    def apply(self, fgraph):
        pass


DEPRECATED_NAMES = [
    (
        "LocalMetaOptimizerSkipAssertionError",
        "`LocalMetaOptimizerSkipAssertionError` is deprecated: use `MetaNodeRewriterSkip` instead.",
        MetaNodeRewriterSkip,
    ),
    (
        "GlobalOptimizer",
        "`GlobalOptimizer` is deprecated: use `GraphRewriter` instead.",
        GraphRewriter,
    ),
    (
        "LocalOptimizer",
        "`LocalOptimizer` is deprecated: use `NodeRewriter` instead.",
        NodeRewriter,
    ),
    (
        "local_optimizer",
        "`local_optimizer` is deprecated: use `node_rewriter` instead.",
        node_rewriter,
    ),
    (
        "pre_greedy_local_optimizer",
        "`pre_greedy_local_optimizer` is deprecated: use `pre_greedy_node_rewriter` instead.",
        pre_greedy_node_rewriter,
    ),
    (
        "FromFunctionOptimizer",
        "`FromFunctionOptimizer` is deprecated: use `FromFunctionGraphRewriter` instead.",
        FromFunctionGraphRewriter,
    ),
    (
        "optimizer",
        "`optimizer` is deprecated: use `graph_rewriter` instead.",
        graph_rewriter,
    ),
    (
        "inplace_optimizer",
        "`inplace_optimizer` is deprecated: use `graph_rewriter` instead.",
        graph_rewriter,
    ),
    (
        "LocalMetaOptimizer",
        "`LocalMetaOptimizer` is deprecated: use `MetaNodeRewriter` instead.",
        MetaNodeRewriter,
    ),
    (
        "SeqOptimizer",
        "`SeqOptimizer` is deprecated: use `SequentialGraphRewriter` instead.",
        SequentialGraphRewriter,
    ),
    (
        "FromFunctionLocalOptimizer",
        "`FromFunctionLocalOptimizer` is deprecated: use `FromFunctionNodeRewriter` instead.",
        FromFunctionNodeRewriter,
    ),
    (
        "LocalOptTracker",
        "`LocalOptTracker` is deprecated: use `OpToRewriterTracker` instead.",
        OpToRewriterTracker,
    ),
    (
        "LocalOptGroup",
        "`LocalOptGroup` is deprecated: use `SequentialNodeRewriter` instead.",
        SequentialNodeRewriter,
    ),
    (
        "OpSub",
        "`OpSub` is deprecated: use `SubstitutionNodeRewriter` instead.",
        SubstitutionNodeRewriter,
    ),
    (
        "OpRemove",
        "`OpRemove` is deprecated: use `RemovalNodeRewriter` instead.",
        RemovalNodeRewriter,
    ),
    (
        "PatternSub",
        "`PatternSub` is deprecated: use `PatternNodeRewriter` instead.",
        PatternNodeRewriter,
    ),
    (
        "NavigatorOptimizer",
        "`NavigatorOptimizer` is deprecated: use `NodeProcessingGraphRewriter` instead.",
        NodeProcessingGraphRewriter,
    ),
    (
        "TopoOptimizer",
        "`TopoOptimizer` is deprecated: use `WalkingGraphRewriter` instead.",
        WalkingGraphRewriter,
    ),
    (
        "topogroup_optimizer",
        "`topogroup_optimizer` is deprecated: use `walking_rewriter` instead.",
        walking_rewriter,
    ),
    (
        "OpKeyOptimizer",
        "`OpKeyOptimizer` is deprecated: use `OpKeyGraphRewriter` instead.",
        OpKeyGraphRewriter,
    ),
    (
        "EquilibriumOptimizer",
        "`EquilibriumOptimizer` is deprecated: use `EquilibriumGraphRewriter` instead.",
        EquilibriumGraphRewriter,
    ),
]


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")

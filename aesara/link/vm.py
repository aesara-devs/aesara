"""
VMs that run Aesara graph computations.

A VM is not actually different from a Linker, we just decided
VM was a better name at some point.

"""
import platform
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import zip_longest
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, Variable
from aesara.link.basic import Container, LocalLinker
from aesara.link.c.exceptions import MissingGXX
from aesara.link.utils import (
    gc_helper,
    get_destroy_dependencies,
    map_storage,
    raise_with_op,
)


if TYPE_CHECKING:
    from aesara.graph.fg import FunctionGraph
    from aesara.graph.op import (
        BasicThunkType,
        ComputeMapType,
        StorageCellType,
        StorageMapType,
    )


def calculate_reallocate_info(
    order: Sequence[Apply],
    fgraph: "FunctionGraph",
    storage_map: "StorageMapType",
    compute_map_re: "ComputeMapType",
    dependencies: Dict[Variable, List[Variable]],
) -> Dict[Variable, List[Variable]]:
    """Finds pairs of computed variables that can share a storage cell.

    This apparently reduces memory allocations, but its scope is very limited
    (e.g. only scalars, only used by the Python VMs without lazy computations).

    Parameters
    ----------
    order
        List of nodes in compute order.
    fgraph
        The `FunctionGraph`.
    storage_map
        Map from variables to their storage cells.
    compute_map_re
        Reallocation map.  TODO
    dependencies
        Map from variables to the variables that depend on them.

    """
    reallocated_info = {}
    viewed_by: Dict[Variable, List[Variable]] = {}
    for var in fgraph.variables:
        viewed_by[var] = []
    view_of: Dict[Variable, Variable] = {}
    pre_allocated = set()
    allocated = set()

    for idx in range(len(order)):
        node = order[idx]
        dmap = node.op.destroy_map
        vmap = node.op.view_map

        idx_o = 0
        for out in node.outputs:
            for var in node.outputs:
                compute_map_re[var][0] = True
            ins = None
            if dmap and idx_o in dmap:
                idx_v = dmap[idx_o]
                assert (
                    len(idx_v) == 1
                ), "Here we only support the possibility to destroy one input"
                ins = node.inputs[idx_v[0]]
            if vmap and idx_o in vmap:
                assert ins is None
                idx_v = vmap[idx_o]
                assert (
                    len(idx_v) == 1
                ), "Here we only support the possibility to view one input"
                ins = node.inputs[idx_v[0]]
            if ins is not None:
                assert isinstance(ins, Variable)
                origin = view_of.get(ins, ins)
                view_of[out] = origin
                viewed_by[origin].append(out)
            idx_o += 1

        for ins in node.inputs:
            assert not (ins in view_of and viewed_by[ins])
            if (
                getattr(ins.type, "ndim", None) == 0
                and not storage_map[ins][0]
                and ins not in fgraph.outputs
                and ins.owner
                and all(compute_map_re[v][0] for v in dependencies.get(ins, []))
                and ins not in allocated
            ):
                # Constant memory cannot be changed
                # Constant and shared variables' storage_map value is not empty
                reuse_out = None
                if ins not in view_of and not viewed_by.get(ins, []):
                    # where gc
                    for i in range(idx + 1, len(order)):
                        if reuse_out is not None:
                            break  # type: ignore
                        for out in order[i].outputs:
                            if (
                                getattr(out.type, "ndim", None) == 0
                                and out not in pre_allocated
                                and out.type.in_same_class(ins.type)
                            ):
                                reuse_out = out
                                pre_allocated.add(out)
                                allocated.add(ins)
                                break
                elif ins in view_of:
                    origin = view_of[ins]
                    if ins in viewed_by[origin]:
                        viewed_by[origin].remove(ins)
                    if (
                        not viewed_by[origin]
                        and origin not in fgraph.inputs
                        and not isinstance(origin, Constant)
                    ):
                        # where gc
                        for i in range(idx + 1, len(order)):
                            if reuse_out is not None:
                                break
                            for out in order[i].outputs:
                                if (
                                    getattr(out.type, "ndim", None) == 0
                                    and out not in pre_allocated
                                    and (out.type.in_same_class(ins.type))
                                ):
                                    reuse_out = out
                                    pre_allocated.add(out)
                                    allocated.add(ins)
                                    break
                if reuse_out is not None:
                    reallocated_info[ins] = [ins, reuse_out]

    return reallocated_info


class VM(ABC):
    r"""An abstract class for evaluating Aesara programs.

    The `VM.__call__` method evaluates an Aesara program.

    `Stack` should be considered the reference `VM`/`Linker` implementation.
    It can correctly evaluate all graphs and is the easiest to read. The `CVM`
    is a port of `Stack` and should have the same behavior, but run faster.
    The `CVM`'s code is harder to read though.

    The other python `VM`\s are perhaps not necessary anymore, and don't take
    advantage of lazy computation, although they still produce the correct
    output for lazy nodes.

    Attributes
    ----------
    call_counts
        List of integers, one for each thunk. ``call_count[i]`` is the number
        of times ``thunks[i]`` was called in the course of computations
        performed by `call_with_timers`.
    call_times
        List of floats, one for each thunk. ``call_times[i]`` is the amount of
        runtime spent on ``thunks[i]`` in the course of computations performed
        by `call_with_timers`.

    need_update_inputs : bool
        ``True`` indicates that `Function.__call__` must implement the feedback
        from output storage to input storage. ``False`` means it *must not*
        repeat that feedback.

    """

    need_update_inputs = True

    def __init__(
        self,
        fgraph: "FunctionGraph",
        nodes: List[Apply],
        thunks: List["BasicThunkType"],
        pre_call_clear: List["StorageCellType"],
    ):
        r"""
        Parameters
        ----------
        fgraph
            The `FunctionGraph` associated with `nodes` and `thunks`.
        nodes
            A list of nodes in toposort order.
        thunks
            A list of thunks to execute those nodes, in toposort order.
        pre_call_clear
            A list of containers to empty at the beginning of each call.
        """

        if len(nodes) != len(thunks):
            raise ValueError("`nodes` and `thunks` must be the same length")

        self.fgraph = fgraph
        self.nodes = nodes
        self.thunks = thunks
        self.pre_call_clear = pre_call_clear
        self.call_counts = [0] * len(nodes)
        self.call_times = [0] * len(nodes)
        self.time_thunks = False
        self.storage_map: Optional[StorageMapType] = None

    @abstractmethod
    def __call__(self):
        r"""Run the virtual machine.

        After this is executed, all the output variables will have been
        computed.  `VM`\s may vary regarding what exactly this means and how it
        is done.
        """

    def clear_storage(self):
        """Free any internal references to temporary variables.

        Essentially, free as much memory as possible without interfering with
        the ability to evaluate subsequent calls.
        """

    def update_profile(self, profile):
        """Update a profile object."""
        for node, thunk, t, c in zip(
            self.nodes, self.thunks, self.call_times, self.call_counts
        ):
            profile.apply_time.setdefault((self.fgraph, node), 0.0)
            profile.apply_time[(self.fgraph, node)] += t

            profile.apply_callcount.setdefault((self.fgraph, node), 0)
            profile.apply_callcount[(self.fgraph, node)] += c

            profile.apply_cimpl[node] = hasattr(thunk, "cthunk")

        if hasattr(self, "variable_shape"):
            profile.variable_shape = self.variable_shape.copy()
            profile.variable_strides = self.variable_strides.copy()
            profile.variable_offset = self.variable_offset.copy()

        if hasattr(self, "node_executed_order"):
            profile.node_executed_order = self.node_executed_order[:]

        if hasattr(self, "node_cleared_order"):
            profile.node_cleared_order = self.node_cleared_order[:]

        if hasattr(self, "dependencies"):
            profile.dependencies = self.dependencies

        # clear the timer info out of the buffers
        for i in range(len(self.call_times)):
            self.call_times[i] = 0.0
            self.call_counts[i] = 0


class UpdatingVM(VM):
    """A `VM` that performs updates on its graph's inputs."""

    need_update_inputs = False

    def __init__(
        self,
        fgraph,
        nodes,
        thunks,
        pre_call_clear,
        storage_map: "StorageMapType",
        input_storage: List["StorageCellType"],
        output_storage: List["StorageCellType"],
        update_vars: Dict[Variable, Variable],
    ):
        r"""
        Parameters
        ----------
        storage_map
            A ``dict`` mapping `Variable`\s to single-element lists where a
            computed value for each `Variable` may be found.
        input_storage
            Storage cells for each input.
        output_storage
            Storage cells for each output.
        update_vars
            A ``dict`` from input to output variables that specify
            output-to-input in-place storage updates that occur after
            evaluation of the entire graph (i.e. all the thunks).
        """
        super().__init__(fgraph, nodes, thunks, pre_call_clear)

        self.storage_map = storage_map
        self.input_storage = input_storage
        self.output_storage = output_storage
        self.inp_storage_and_out_idx = tuple(
            (inp_storage, self.fgraph.outputs.index(update_vars[inp]))
            for inp, inp_storage in zip(self.fgraph.inputs, self.input_storage)
            if inp in update_vars
        )

    def perform_updates(self) -> List[Any]:
        """Perform the output-to-input updates and return the output values."""

        # The outputs need to be collected *before* the updates that follow
        outputs = [cell[0] for cell in self.output_storage]

        for inp_storage, out_idx in self.inp_storage_and_out_idx:
            inp_storage[0] = outputs[out_idx]

        return outputs


class Loop(UpdatingVM):
    """Unconditional start-to-finish program execution in Python.

    Garbage collection is possible on intermediate results when the
    `post_thunk_clear` constructor argument is non-``None``.
    """

    def __init__(
        self,
        fgraph,
        nodes,
        thunks,
        pre_call_clear,
        storage_map,
        input_storage,
        output_storage,
        update_vars,
        post_thunk_clear: Optional[List["StorageCellType"]] = None,
    ):
        r"""
        Parameters
        ----------
        post_thunk_clear
            A list of storage cells for each thunk that should be cleared after
            each thunk is evaluated.  This is the "garbage collection"
            functionality.
        """
        super().__init__(
            fgraph,
            nodes,
            thunks,
            pre_call_clear,
            storage_map,
            input_storage,
            output_storage,
            update_vars,
        )

        if post_thunk_clear is not None:
            if not (len(nodes) == len(thunks) == len(post_thunk_clear)):
                raise ValueError(
                    "`nodes`, `thunks` and `post_thunk_clear` are not the same lengths"
                )
            # Some other part of Aesara use this information
            self.allow_gc = True
            self.post_thunk_clear = post_thunk_clear
        else:
            self.allow_gc = False
            self.post_thunk_clear = []

    def __call__(self):
        if self.time_thunks:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                i = 0
                for thunk, node, old_storage in zip_longest(
                    self.thunks, self.nodes, self.post_thunk_clear, fillvalue=()
                ):
                    t0 = time.perf_counter()
                    thunk()
                    t1 = time.perf_counter()
                    self.call_counts[i] += 1
                    self.call_times[i] += t1 - t0
                    for old_s in old_storage:
                        old_s[0] = None
                    i += 1
            except Exception:
                raise_with_op(self.fgraph, node, thunk)
        else:
            for cont in self.pre_call_clear:
                cont[0] = None
            try:
                for thunk, node, old_storage in zip_longest(
                    self.thunks, self.nodes, self.post_thunk_clear, fillvalue=()
                ):
                    thunk()
                    for old_s in old_storage:
                        old_s[0] = None
            except Exception:
                raise_with_op(self.fgraph, node, thunk)

        return self.perform_updates()


class Stack(UpdatingVM):
    """Finish-to-start evaluation order of thunks.

    This supports lazy evaluation of subtrees and partial computations of
    graphs when only some inputs have changed.

    At a pseudo-code level, the basic idea is as follows:

    .. code-block:: python

        def recursively_evaluate(var):
            if var is up to date:
                return
            if var.owner.inputs are up to date:
                update var
                return
            for input in var.owner.inputs:
                recursively_evaluate(var)

        for output in outputs:
            recursively_evaluate(output)


    The actual logic is more complex to support intermediate garbage
    collection, lazily-evaluated nodes, and better speed.

    """

    def __init__(
        self,
        fgraph,
        nodes,
        thunks,
        pre_call_clear,
        storage_map,
        input_storage,
        output_storage,
        update_vars,
        compute_map: "ComputeMapType",
        allow_gc: bool,
        dependencies: Optional[Dict[Variable, List[Variable]]] = None,
        callback=None,
        callback_input=None,
    ):
        r"""
        Parameters
        ----------
        allow_gc
            Determines whether or not garbage collection is performed.
        dependencies
            TODO
        callback
            TODO
        callback_input
            TODO
        """

        super().__init__(
            fgraph,
            nodes,
            thunks,
            pre_call_clear,
            storage_map,
            input_storage,
            output_storage,
            update_vars,
        )

        self.update_vars = update_vars
        self.compute_map = compute_map
        self.allow_gc = allow_gc
        self.message = ""
        self.base_apply_stack = [o.owner for o in fgraph.outputs if o.owner]
        self.outputs = fgraph.outputs
        self.variable_shape: Dict[Variable, Any] = {}  # Variable -> shape
        self.variable_strides: Dict[Variable, Any] = {}  # Variable -> strides
        self.variable_offset: Dict[Variable, Any] = {}  # Variable -> offset
        node_idx = {node: i for i, node in enumerate(self.nodes)}
        self.node_idx = node_idx
        self.callback = callback
        self.callback_input = callback_input
        self.destroy_dependencies = get_destroy_dependencies(fgraph)
        self.dependencies = dependencies

        if self.allow_gc and self.dependencies is None:
            raise ValueError("Must set dependencies when using GC")

    def run_thunk_of_node(self, node):
        """
        Run the thunk corresponding to Apply instance `node`.

        Calls self.callback if it is defined.

        """
        idx = self.node_idx[node]
        t0 = time.perf_counter()
        rval = self.thunks[idx]()
        self.node_executed_order.append(node)

        # Some thunks on some computers run faster than the granularity
        # of the time.perf_counter clock.
        # Profile output looks buggy if a node has run but takes 0 time.
        # (and profile code might hide real bugs if it rounds up 0)
        dt = max(time.perf_counter() - t0, 1e-10)
        if self.callback is not None:
            self.callback(
                node=node,
                thunk=self.thunks[idx],
                storage_map=self.storage_map,
                compute_map=self.compute_map,
            )
        return rval, dt

    def __call__(self, output_subset=None):
        storage_map = self.storage_map
        compute_map = self.compute_map
        thunks = self.thunks
        dependencies = self.dependencies
        self.node_executed_order = []
        self.node_cleared_order = []

        for cont in self.pre_call_clear:
            cont[0] = None

        for k in self.storage_map:
            compute_map[k][0] = k.owner is None
            if self.callback_input and compute_map[k][0]:
                self.callback_input(k, self.storage_map[k][0])

        # apply_stack contains nodes
        if output_subset is not None:
            # Add the outputs that are needed for the in-place updates of the
            # inputs in `self.update_vars`
            output_subset = list(output_subset)
            for inp, out in self.update_vars.items():
                out_idx = self.fgraph.outputs.index(out)
                if out_idx not in output_subset:
                    output_subset.append(out_idx)

            apply_stack = [
                self.outputs[i].owner for i in output_subset if self.outputs[i].owner
            ]
        else:
            apply_stack = list(self.base_apply_stack)

        last_apply_stack_len = -1

        # This record all function inputs/shared variables and constants
        for var, data in self.storage_map.items():
            if data[0] is None:
                continue
            if hasattr(var.type, "get_shape_info"):
                sh = var.type.get_shape_info(data[0])
            else:
                sh = "no shape"
            self.variable_shape[var] = sh
            st = getattr(data[0], "strides", "no strides")
            if getattr(data[0], "flags", False) and data[0].flags.c_contiguous:
                st = "c"
            elif hasattr(data[0], "is_c_contiguous") and data[0].is_c_contiguous():
                st = "c"
            self.variable_strides[var] = st
            off = getattr(data[0], "offset", "")
            self.variable_offset[var] = off

        while apply_stack:
            # Make sure something happened last time round.  This is
            # just a safety check to make sure the op is written
            # correctly apply_stack should either decrease in length
            # by one (a thunk successfully applied), or increase in
            # length (added dependencies over and above the original).
            # NB: this doesn't catch cycles (would be too expensive/slow),
            #     just stalls.
            apply_stack_len = len(apply_stack)
            assert apply_stack_len != last_apply_stack_len
            last_apply_stack_len = apply_stack_len

            current_apply = apply_stack.pop()
            current_inputs = current_apply.inputs
            current_outputs = current_apply.outputs
            current_deps = current_inputs + self.destroy_dependencies[current_apply]

            computed_ins = all(compute_map[v][0] for v in current_deps)
            computed_outs = all(compute_map[v][0] for v in current_outputs)

            if not thunks[self.node_idx[current_apply]].lazy:
                #
                # stack loop: Normal Non-Lazy Case
                # ================================
                #
                # Check if all inputs are in place
                # If so compute thunk and remove it from the apply_stack
                # If not leave it in, and add to the apply_stack those
                # that will produce you those inputs

                if computed_ins and not computed_outs:
                    # -- Non-lazy case: have inputs, time to compute outputs
                    try:
                        _, dt = self.run_thunk_of_node(current_apply)
                        del _
                        if config.profile or config.print_global_stats:
                            current_idx = self.node_idx[current_apply]
                            self.call_counts[current_idx] += 1
                            self.call_times[current_idx] += dt
                            # Computing the memory footprint of the the op
                            # ?? What about inplace .. if the op is inplace
                            # you don't actually ask for more memory!
                            for (idx, o) in enumerate(
                                thunks[self.node_idx[current_apply]].outputs
                            ):
                                var = self.nodes[current_idx].outputs[idx]
                                if hasattr(var.type, "get_shape_info"):
                                    sh = var.type.get_shape_info(o[0])
                                else:
                                    sh = "no shape"
                                self.variable_shape[var] = sh
                                st = getattr(o[0], "strides", "no strides")
                                if (
                                    getattr(o[0], "flags", False)
                                    and o[0].flags.c_contiguous
                                ):
                                    st = "c"
                                elif (
                                    hasattr(o[0], "is_c_contiguous")
                                    and o[0].is_c_contiguous()
                                ):
                                    st = "c"
                                self.variable_strides[var] = st
                                off = getattr(o[0], "offset", "")
                                self.variable_offset[var] = off
                    except Exception:
                        raise_with_op(
                            self.fgraph,
                            current_apply,
                            self.thunks[self.node_idx[current_apply]],
                            storage_map=storage_map,
                        )
                    for o in current_apply.outputs:
                        compute_map[o][0] = 1

                    input_index = []
                    # A list store the index of inputs variables

                    if self.allow_gc:
                        for i in current_apply.inputs:
                            # Garbage Collection -> check if anybody else uses
                            # this input
                            if dependencies[i] and i.owner and i not in self.outputs:
                                if all(compute_map[v][0] for v in dependencies[i]):
                                    storage_map[i][0] = None
                                    input_index.append(current_apply.inputs.index(i))

                                    # DO NOT set compute_map to 0

                                    # If values become False and the
                                    # current_apply is still in the
                                    # stack, this will cause it to be
                                    # recomputed! This can cause wrong value
                                    # with some combination of inplace op.
                                    compute_map[i][0] = 2
                    self.node_cleared_order.append(input_index)

                elif not computed_ins:
                    # -- Non-lazy case, need inputs
                    apply_stack.append(current_apply)
                    apply_stack.extend(inp.owner for inp in current_deps if inp.owner)

            elif not computed_outs:
                #
                # stack loop: Lazy Evaluation Case
                # ================================
                #
                # Lazy evaluation protocol is to run the thunk with the
                # current storage_map and compute_map accessed via closure,
                # and the thunk will return a list of variables from its input
                # list that it requires.

                try:
                    requires, dt = self.run_thunk_of_node(current_apply)
                    current_idx = self.node_idx[current_apply]
                    self.call_counts[current_idx] += 1
                    self.call_times[current_idx] += dt

                except Exception:
                    raise_with_op(
                        self.fgraph,
                        current_apply,
                        self.thunks[self.node_idx[current_apply]],
                        storage_map=storage_map,
                    )

                if requires:
                    for r in requires:
                        # We are not done with this op ..  so we added
                        # back and see to get the inputs we are
                        # missing
                        apply_stack.append(current_apply)
                        if current_apply.inputs[r].owner:
                            apply_stack.append(current_apply.inputs[r].owner)
                else:
                    if config.profile or config.print_global_stats:
                        for (idx, o) in enumerate(
                            thunks[self.node_idx[current_apply]].outputs
                        ):
                            var = self.nodes[self.node_idx[current_apply]].outputs[idx]

                            if hasattr(var.type, "get_shape_info"):
                                sh = var.type.get_shape_info(o[0])
                            else:
                                sh = "no shape"
                            self.variable_shape[var] = sh
                            st = getattr(o[0], "strides", "no strides")
                            if (
                                getattr(o[0], "flags", False)
                                and o[0].flags.c_contiguous
                            ):
                                st = "c"
                            elif (
                                hasattr(o[0], "is_c_contiguous")
                                and o[0].is_c_contiguous()
                            ):
                                st = "c"
                            self.variable_strides[var] = st
                            off = getattr(o[0], "offset", "")
                            self.variable_offset[var] = off

                    input_index = []

                    if self.allow_gc:
                        for i in current_apply.inputs:
                            if dependencies[i] and i.owner and i not in self.outputs:
                                empty_storage_map = True
                                for x in dependencies[i]:
                                    if not compute_map[x][0]:
                                        empty_storage_map = False
                                        break
                                if empty_storage_map:
                                    storage_map[i][0] = None
                                    input_index.append(current_apply.inputs.index(i))
                                    # See the not lazy gc code for explanations
                                    # of compute_map change
                                    compute_map[i][0] = 2

                    self.node_cleared_order.append(input_index)

        # Hacky coarse gc final pass
        # This is required until we have a proper gc algorithm for graphs with
        # lazy evaluation. See discussion on theano-dev June 19 2012.
        final_index = []

        if self.allow_gc:
            for v in storage_map:
                if v.owner and v not in self.outputs:
                    if compute_map[v][0] == 2:
                        continue
                    else:
                        storage_map[v][0] = None
                        final_index.append(v)
                        compute_map[v][0] = 2

        self.node_cleared_order.append(final_index)

        return self.perform_updates()


class VMLinker(LocalLinker):
    """Class that satisfies the `Linker` interface by acting as a `VM` factory.

    Parameters
    ----------
    allow_gc
        Force the virtual machine to clean up unnecessary references, in order
        to allow garbage collection on intermediate values during computation
        of a function.  If ``None``, use as default the value of the Aesara
        flag `allow_gc`.
    use_cloop
        Use the C-based virtual machine if possible
    callback
        A callable object to call after each call to a thunk within the virtual
        machine.  It will be called with four arguments: ``node``, ``thunk``,
        ``storage_map``, and ``compute_map``.
    callback_input
        A callable object to call on each input to the graph (variables with no
        owner).  This includes constants and shared variables values.  It will
        be called with two arguments: ``var``, ``value``.
    lazy
        Useful only when `use_cloop` is False. When `lazy` is ``None``, use the
        Aesara flag ``vm__lazy`` value. Then if we have a ``None`` (default) we
        auto detect if lazy evaluation is needed and use the appropriate
        version. If `lazy` is ``True`` or ``False``, we force the version used
        between `Loop` and `Stack`.
    c_thunks
        If ``None`` or ``True``, don't change the default. If ``False``, don't
        compile C code for the thunks.
    allow_partial_eval
        If ``True``, enforces usage of `Stack` or `CVM`, to allow for partial
        evaluation of functions (calculating a subset of outputs).

    """

    def __init__(
        self,
        allow_gc=None,
        use_cloop=False,
        callback=None,
        callback_input=None,
        lazy=None,
        schedule=None,
        c_thunks=None,
        allow_partial_eval=None,
    ):
        # Note: if more parameters are added to __init__, make sure to forward
        # them in the "type(self)(...)" call in the "accept" method below.
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph = None
        self.use_cloop = use_cloop
        self.callback = callback
        self.callback_input = callback_input
        self.lazy = lazy
        if c_thunks is None:
            c_thunks = bool(config.cxx)
        self.c_thunks = c_thunks
        self.allow_partial_eval = allow_partial_eval
        self.updated_vars = {}
        super().__init__(allow_gc=allow_gc, scheduler=schedule)

    def accept(self, fgraph, no_recycling=None, profile=None):
        """Check if fgraph is the first FunctionGraph that has ever been
        associated to self, else, create a new `VMLinker`
        associated to fgraph

        Parameters
        ----------
        fgraph
            A PerformLinker can have accepted one FunctionGraph instance
            at a time.

        no_recycling

            no_recycling is a list of storage (list of 1 element, the
            value corresponding to one variable). Those variable
            storage should not be reused after the call that created
            them.

            This happen for example for output of the graph that we
            give to the user. We don't want to reuse those object in
            case the user have kept it.

            `VMLinker` make sure this happen by setting the list
            element to None at the start of each call.

            Older Linker use not exactly the same mechanism. They will
            also modify the c code to don't look up the value in the
            storage. This cause duplicate c code compilation for the
            same op if they are in the middle of the graph or in the
            no_recycling. We don't want that, so compile all c code
            the same (middle of the graph vs output).

            TODO: change the logic to remove the reference at the end
            of the call instead of the start. This will request all VM
            implementation (Loop, Stack, CVM).__call__ to
            return the user outputs as Function.__call__ won't be able
            to find them anymore.

        Returns
        -------
        Self if fgraph is the first FunctionGraph that has ever been
        associated to self, else, a new `VMLinker` associated to fgraph.

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            # Build a new `VMLinker`, and call accept on that one.
            # Warning: make sure to forward the correct values of
            # all parameters to __init__ here.
            return type(self)(
                allow_gc=self.allow_gc,
                use_cloop=self.use_cloop,
                callback=self.callback,
                callback_input=self.callback_input,
                lazy=self.lazy,
                schedule=self.schedule,
                c_thunks=self.c_thunks,
                allow_partial_eval=self.allow_partial_eval,
            ).accept(fgraph, no_recycling, profile)
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        self.profile = profile

        return self

    def accept_var_updates(self, updated_vars):
        """Records in the `Linker` which variables have update expressions.

        It does not imply that the `Linker` will actually implement these updates
        (see `VM.need_update_inputs`).  This mechanism is admittedly confusing, and
        it could use some cleaning up. The base `Linker` object should probably
        go away completely.

        TODO: Remove this after refactoring the `VM`/`Linker` interfaces.
        """
        self.updated_vars = updated_vars

    def compute_gc_dependencies(self, variables):
        """
        Returns dict: variable K -> list of variables [v1, v2, v3, ...]
        for each K in variables.

        The variables v1, v2, ... are the full set of variables that depend
        directly on K. When we know that none of them will need to be
        computed, we know that:
        * K will not need to be computed.
        * If K is already computed, it can be released for garbage collection.

        Parameters
        ----------
        variables
            Iterable over the variables used in a graph computation.

        Notes
        -----
        It doesn't take care of the view_map/destroy_map. So it means it relies
        on Python gc no to free the object real storage.

        N.B. gc means garbage collection

        """
        dependencies = {}
        for k in variables:
            dependencies[k] = []
            # If k has no owner, it is an input / constant and its value
            # should not be removed from the storage_map because we have no
            # way of getting it back.
            #
            # XXX if k has no clients... what is it doing in the computation?
            # Fred guess: it could happen for node with multiple outputs when
            # we don't use all outputs.

            if k.owner and self.fgraph.clients[k]:
                ls = []
                for cl in self.fgraph.clients[k]:
                    if cl[0] != "output":
                        ls += cl[0].outputs
                dependencies[k] += ls
        return dependencies

    def reduce_storage_allocations(
        self, storage_map: "StorageMapType", order: Sequence[Apply]
    ) -> Tuple[Variable, ...]:
        """Reuse storage cells in a storage map.

        `storage_map` is updated in-place.

        When this feature is used, `storage_map` will no longer have a
        one-to-one mapping with the original variables, because--for
        example--some outputs may share storage with intermediate values.

        Returns
        -------
        A tuple of the variables that were reallocated.

        """
        # Collect Reallocation Info
        compute_map_re: DefaultDict[Variable, List[bool]] = defaultdict(lambda: [False])
        for var in self.fgraph.inputs:
            compute_map_re[var][0] = True

        if getattr(self.fgraph.profile, "dependencies", None):
            dependencies = self.fgraph.profile.dependencies
        else:
            dependencies = self.compute_gc_dependencies(storage_map)

        reallocated_info: Dict[Variable, List[Variable]] = calculate_reallocate_info(
            order, self.fgraph, storage_map, compute_map_re, dependencies
        )
        for pair in reallocated_info.values():
            storage_map[pair[1]] = storage_map[pair[0]]

        return tuple(reallocated_info.keys())

    def make_vm(
        self,
        nodes,
        thunks,
        input_storage,
        output_storage,
        storage_map,
        post_thunk_clear,
        computed,
        compute_map,
        updated_vars,
    ):

        pre_call_clear = [storage_map[v] for v in self.no_recycling]

        try:
            from aesara.link.c.cvm import CVM
        except (MissingGXX, ImportError):
            CVM = None

        if (
            self.callback is not None
            or self.callback_input is not None
            or ((config.profile or config.print_global_stats) and config.profile_memory)
            or (self.allow_partial_eval and not self.use_cloop)
        ):

            if self.use_cloop and (
                self.callback is not None or self.callback_input is not None
            ):
                warnings.warn("CVM does not support callback, using Stack VM.")
            if self.use_cloop and config.profile_memory:
                warnings.warn("CVM does not support memory profiling, using Stack VM.")
            if not self.use_cloop and self.allow_partial_eval:
                warnings.warn(
                    "Loop VM does not support partial evaluation, using Stack VM."
                )
            # Needed for allow_gc=True, profiling and storage_map reuse
            deps = self.compute_gc_dependencies(storage_map)
            vm = Stack(
                self.fgraph,
                nodes,
                thunks,
                pre_call_clear,
                storage_map,
                input_storage,
                output_storage,
                updated_vars,
                compute_map,
                self.allow_gc,
                dependencies=deps,
                callback=self.callback,
                callback_input=self.callback_input,
            )
        elif self.use_cloop and CVM:

            # create a map from nodes to ints and vars to ints
            nodes_idx = {}
            vars_idx = {}
            for i, node in enumerate(nodes):
                nodes_idx[node] = i
                for v in node.inputs + node.outputs:
                    vars_idx.setdefault(v, len(vars_idx))
            for v in self.fgraph.inputs + self.fgraph.outputs:
                vars_idx.setdefault(v, len(vars_idx))

            nodes_idx_inv = {}
            vars_idx_inv = {}
            for (node, i) in nodes_idx.items():
                nodes_idx_inv[i] = node
            for (var, i) in vars_idx.items():
                vars_idx_inv[i] = var

            # put storage_map and compute_map into a int-based scheme
            storage_map_list = [
                storage_map[vars_idx_inv[i]] for i in range(len(vars_idx_inv))
            ]
            compute_map_list = [
                compute_map[vars_idx_inv[i]] for i in range(len(vars_idx_inv))
            ]
            if nodes:
                assert type(storage_map_list[0]) is list
                assert type(compute_map_list[0]) is list

            # Needed for allow_gc=True, profiling and storage_map reuse
            dependency_map = self.compute_gc_dependencies(storage_map)
            dependency_map_list = [
                [vars_idx[d] for d in dependency_map[vars_idx_inv[i]]]
                for i in range(len(vars_idx_inv))
            ]

            # build the pointers to node inputs and offsets
            base_input_output_list = []
            node_n_inputs = []
            node_n_outputs = []
            node_input_offset = []
            node_output_offset = []
            for node in nodes:
                inputs_idx = [vars_idx[v] for v in node.inputs]
                outputs_idx = [vars_idx[v] for v in node.outputs]
                node_n_inputs.append(len(inputs_idx))
                node_n_outputs.append(len(outputs_idx))
                node_input_offset.append(len(base_input_output_list))
                base_input_output_list.extend(inputs_idx)
                node_output_offset.append(len(base_input_output_list))
                base_input_output_list.extend(outputs_idx)

            # build the var owner array
            var_owner = [None] * len(vars_idx)
            for (var, i) in vars_idx.items():
                if var.owner:
                    var_owner[i] = nodes_idx[var.owner]

            is_lazy_list = [int(th.lazy) for th in thunks]
            output_vars = [vars_idx[v] for v in self.fgraph.outputs]

            # builds the list of prereqs induced by e.g. destroy_handler
            ords = self.fgraph.orderings()
            node_prereqs = []
            node_output_size = []
            for i, node in enumerate(nodes):
                node_output_size.append(0)
                prereq_var_idxs = []
                for prereq_node in ords.get(node, []):
                    prereq_var_idxs.extend([vars_idx[v] for v in prereq_node.outputs])
                prereq_var_idxs = list(set(prereq_var_idxs))
                prereq_var_idxs.sort()  # TODO: why sort?
                node_prereqs.append(prereq_var_idxs)

            # This is essentially a version of `self.fgraph.update_mapping`.
            # It specifies the outputs-to-inputs updates via the pairs
            # `(input_idx, output_idx)` (i.e. the input at index `input_idx`
            # takes the value of the output at index `output_idx`).
            update_storage = tuple(
                (vars_idx[in_var], self.fgraph.outputs.index(out_var))
                for in_var, out_var in updated_vars.items()
            )

            # PyPy has no sys.getrefcount, so ignore this check if not running
            # under CPython.
            if platform.python_implementation() == "CPython":
                c0 = sys.getrefcount(node_n_inputs)

            vm = CVM(
                self.fgraph,
                nodes,
                thunks,
                pre_call_clear,
                allow_gc=self.allow_gc,
                call_counts=[0] * len(nodes),
                call_times=[0.0] * len(nodes),
                compute_map_list=compute_map_list,
                storage_map_list=storage_map_list,
                base_input_output_list=base_input_output_list,
                node_n_inputs=node_n_inputs,
                node_n_outputs=node_n_outputs,
                node_input_offset=node_input_offset,
                node_output_offset=node_output_offset,
                var_owner=var_owner,
                is_lazy_list=is_lazy_list,
                output_vars=output_vars,
                node_prereqs=node_prereqs,
                node_output_size=node_output_size,
                update_storage=update_storage,
                dependencies=dependency_map_list,
            )

            if platform.python_implementation() == "CPython" and c0 != sys.getrefcount(
                node_n_inputs
            ):
                warnings.warn(
                    "Detected reference count inconsistency after CVM construction"
                )
        else:
            lazy = self.lazy
            if lazy is None:
                lazy = config.vm__lazy
            if lazy is None:
                lazy = any(th.lazy for th in thunks)
            if not lazy:
                # there is no conditional in the graph
                vm = Loop(
                    self.fgraph,
                    nodes,
                    thunks,
                    pre_call_clear,
                    storage_map,
                    input_storage,
                    output_storage,
                    updated_vars,
                    post_thunk_clear if self.allow_gc else None,
                )
            else:
                # Needed when allow_gc=True and profiling
                deps = self.compute_gc_dependencies(storage_map)
                vm = Stack(
                    self.fgraph,
                    nodes,
                    thunks,
                    pre_call_clear,
                    storage_map,
                    input_storage,
                    output_storage,
                    updated_vars,
                    compute_map,
                    self.allow_gc,
                    dependencies=deps,
                )
        return vm

    def make_all(
        self,
        profiler=None,
        input_storage=None,
        output_storage=None,
        storage_map=None,
    ):
        fgraph = self.fgraph
        order = self.schedule(fgraph)

        input_storage, output_storage, storage_map = map_storage(
            fgraph, order, input_storage, output_storage, storage_map
        )
        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []

        t0 = time.perf_counter()
        linker_make_thunk_time = {}
        impl = None
        if self.c_thunks is False:
            impl = "py"
        for node in order:
            try:
                thunk_start = time.perf_counter()
                # no-recycling is done at each VM.__call__ So there is
                # no need to cause duplicate c code by passing
                # no_recycling here.
                thunks.append(
                    node.op.make_thunk(node, storage_map, compute_map, [], impl=impl)
                )
                linker_make_thunk_time[node] = time.perf_counter() - thunk_start
                if not hasattr(thunks[-1], "lazy"):
                    # We don't want all ops maker to think about lazy Ops.
                    # So if they didn't specify that its lazy or not, it isn't.
                    # If this member isn't present, it will crash later.
                    thunks[-1].lazy = False
            except Exception:
                raise_with_op(fgraph, node)

        t1 = time.perf_counter()

        if self.profile:
            self.profile.linker_node_make_thunks += t1 - t0
            self.profile.linker_make_thunk_time = linker_make_thunk_time

        for node, thunk in zip(order, thunks):
            thunk.inputs = [storage_map[v] for v in node.inputs]
            thunk.outputs = [storage_map[v] for v in node.outputs]

        lazy = self.lazy
        if lazy is None:
            lazy = config.vm__lazy
        if lazy is None:
            lazy = any(th.lazy for th in thunks)
        if not (
            lazy
            or ((config.profile or config.print_global_stats) and config.profile_memory)
            or self.use_cloop
            or self.callback
            or self.callback_input
        ):
            reallocated_vars = self.reduce_storage_allocations(storage_map, order)
        else:
            reallocated_vars = ()

        computed, last_user = gc_helper(order)
        if self.allow_gc:
            post_thunk_clear = []
            for node in order:
                clear_after_this_thunk = []
                for input in node.inputs:
                    if (
                        input in computed
                        and input not in fgraph.outputs
                        and node == last_user[input]
                        and input not in reallocated_vars
                    ):
                        clear_after_this_thunk.append(storage_map[input])
                post_thunk_clear.append(clear_after_this_thunk)
        else:
            post_thunk_clear = None

        vm = self.make_vm(
            order,
            thunks,
            input_storage,
            output_storage,
            storage_map,
            post_thunk_clear,
            computed,
            compute_map,
            self.updated_vars,
        )

        vm.storage_map = storage_map
        vm.compute_map = compute_map

        return (
            vm,
            [
                Container(input, storage)
                for input, storage in zip(fgraph.inputs, input_storage)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage)
            ],
            thunks,
            order,
        )

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "c_thunks"):
            self.c_thunks = True
        if not hasattr(self, "allow_partial_eval"):
            self.allow_partial_eval = None
        if not hasattr(self, "callback_input"):
            self.callback_input = None

    def __repr__(self):
        args_str = ", ".join(
            [
                f"{name}={getattr(self, name)}"
                for name in ("use_cloop", "lazy", "allow_partial_eval", "allow_gc")
            ]
        )
        return f"{type(self).__name__}({args_str})"

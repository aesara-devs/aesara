"""Pretty-printing (pprint()), the 'Print' Op, debugprint() and pydotprint().

They all allow different way to print a graph or the result of an Op
in a graph(Print Op)
"""

import hashlib
import logging
import os
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import copy
from functools import reduce, singledispatch
from io import IOBase, StringIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from aesara.compile import Function, SharedVariable
from aesara.compile.io import In, Out
from aesara.compile.profiling import ProfileStats
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, Variable, graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import HasInnerGraph, Op, StorageMapType
from aesara.graph.utils import Scratchpad


pydot_imported = False
pydot_imported_msg = ""
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pd

    if pd.find_graphviz():
        pydot_imported = True
    else:
        pydot_imported_msg = "pydot-ng can't find graphviz. Install graphviz."
except ImportError:
    try:
        # fall back on pydot if necessary
        import pydot as pd

        if hasattr(pd, "find_graphviz"):
            if pd.find_graphviz():
                pydot_imported = True
            else:
                pydot_imported_msg = "pydot can't find graphviz"
        else:
            pd.Dot.create(pd.Dot())
            pydot_imported = True
    except ImportError:
        # tests should not fail on optional dependency
        pydot_imported_msg = (
            "Install the python package pydot or pydot-ng." " Install graphviz."
        )
    except Exception as e:
        pydot_imported_msg = "An error happened while importing/trying pydot: "
        pydot_imported_msg += str(e.args)


_logger = logging.getLogger("aesara.printing")
VALID_ASSOC = {"left", "right", "either"}


def char_from_number(number):
    """Convert numbers to strings by rendering it in base 26 using capital letters as digits."""

    base = 26

    rval = ""

    if number == 0:
        rval = "A"

    while number != 0:
        remainder = number % base
        new_char = chr(ord("A") + remainder)
        rval = new_char + rval
        number //= base

    return rval


@singledispatch
def op_debug_information(op: Op, node: Apply) -> Dict[Apply, Dict[Variable, str]]:
    """Provide extra debug print information based on the type of `Op` and `Apply` node.

    Implementations of this dispatch function should return a ``dict`` keyed by
    the `Apply` node, `node`, associated with the given `op`.  The value
    associated with the `node` is another ``dict`` mapping `Variable` inputs
    and/or outputs of `node` to their debug information.

    The `node` key allows the information in the ``dict``'s values to be
    specific to the given `node`, so that--for instance--the provided debug
    information is only ever printed/associated with a given `Variable`
    input/output when that `Variable` is displayed as an input/output of `node`
    and not in every/any other place where said `Variable` is present in a
    graph.

    """
    return {}


def debugprint(
    obj: Union[
        Union[Variable, Apply, Function], List[Union[Variable, Apply, Function]]
    ],
    depth: int = -1,
    print_type: bool = False,
    file: Optional[Union[str, IOBase]] = None,
    ids: str = "CHAR",
    stop_on_name: bool = False,
    done: Optional[Dict[Apply, str]] = None,
    print_storage: bool = False,
    used_ids: Optional[Dict[Variable, str]] = None,
    print_op_info: bool = False,
    print_destroy_map: bool = False,
    print_view_map: bool = False,
    print_fgraph_inputs: bool = False,
) -> Union[str, IOBase]:
    r"""Print a computation graph as text to stdout or a file.

    Each line printed represents a Variable in the graph.
    The indentation of lines corresponds to its depth in the symbolic graph.
    The first part of the text identifies whether it is an input
    (if a name or type is printed) or the output of some Apply (in which case
    the Op is printed).
    The second part of the text is an identifier of the Variable.
    If print_type is True, we add a part containing the type of the Variable

    If a Variable is encountered multiple times in the depth-first search,
    it is only printed recursively the first time. Later, just the Variable
    identifier is printed.

    If an Apply has multiple outputs, then a '.N' suffix will be appended
    to the Apply's identifier, to indicate which output a line corresponds to.

    Parameters
    ----------
    obj
        The `Variable`, `Apply`, or `Function` instance to print (or a list
        thereof).
    depth
        Print graph to this depth (``-1`` for unlimited).
    print_type
        Whether to print the type of printed objects
    file
        When `file` is extends `IOBase`, print to this file; when `file` is
        equal to ``"str"``, return a string; when `file` is ``None``, print to
        stdout.
    ids
        Determines the type of identifier used for variables.
          - ``"id"``: print the python id value,
          - ``"int"``: print integer character,
          - ``"CHAR"``: print capital character,
          - ``"auto"``: print the ``auto_name`` value,
          - ``""``: don't print an identifier.
    stop_on_name
        When ``True``, if a node in the graph has a name, we don't print anything
        below it.
    done
        A ``dict`` where we store the ids of printed nodes.
        Useful to have multiple call to `debugprint` share the same ids.
    print_storage
        If ``True``, this will print the storage map for Aesara functions. When
        combined with ``allow_gc=False``, after the execution of an Aesara
        function, the output will show the intermediate results.
    used_ids
        A map between nodes and their printed ids.
    print_op_info
        Print extra information provided by the relevant `Op`\s.  For example,
        print the tap information for `Scan` inputs and outputs.
    print_destroy_map
        Whether to print the `destroy_map`\s of printed objects
    print_view_map
        Whether to print the `view_map`\s of printed objects
    print_fgraph_inputs
        Print the inputs of `FunctionGraph`\s.

    Returns
    -------
    A string representing the printed graph, if `file` is a string, else `file`.

    """
    if not isinstance(depth, int):
        raise Exception("depth parameter must be an int")

    if file == "str":
        _file = StringIO()
    elif file is None:
        _file = sys.stdout
    else:
        _file = file

    if done is None:
        done = dict()

    if used_ids is None:
        used_ids = dict()

    inputs_to_print = []
    outputs_to_print = []
    profile_list: List[Optional[Any]] = []
    order: List[Optional[List[Apply]]] = []  # Toposort
    smap: List[Optional[StorageMapType]] = []  # storage_map

    if isinstance(obj, (list, tuple, set)):
        lobj = obj
    else:
        lobj = [obj]

    for obj in lobj:
        if isinstance(obj, Variable):
            outputs_to_print.append(obj)
            profile_list.append(None)
            smap.append(None)
            order.append(None)
        elif isinstance(obj, Apply):
            outputs_to_print.extend(obj.outputs)
            profile_list.extend([None for item in obj.outputs])
            smap.extend([None for item in obj.outputs])
            order.extend([None for item in obj.outputs])
        elif isinstance(obj, Function):
            if print_fgraph_inputs:
                inputs_to_print.extend(obj.maker.fgraph.inputs)
            outputs_to_print.extend(obj.maker.fgraph.outputs)
            profile_list.extend([obj.profile for item in obj.maker.fgraph.outputs])
            if print_storage:
                smap.extend([obj.vm.storage_map for item in obj.maker.fgraph.outputs])
            else:
                smap.extend([None for item in obj.maker.fgraph.outputs])
            topo = obj.maker.fgraph.toposort()
            order.extend([topo for item in obj.maker.fgraph.outputs])
        elif isinstance(obj, FunctionGraph):
            if print_fgraph_inputs:
                inputs_to_print.extend(obj.inputs)
            outputs_to_print.extend(obj.outputs)
            profile_list.extend([getattr(obj, "profile", None) for item in obj.outputs])
            smap.extend([getattr(obj, "storage_map", None) for item in obj.outputs])
            topo = obj.toposort()
            order.extend([topo for item in obj.outputs])
        elif isinstance(obj, (int, float, np.ndarray)):
            print(obj, file=_file)
        elif isinstance(obj, (In, Out)):
            outputs_to_print.append(obj.variable)
            profile_list.append(None)
            smap.append(None)
            order.append(None)
        else:
            raise TypeError(f"debugprint cannot print an object type {type(obj)}")

    inner_graph_ops = []
    if any(p for p in profile_list if p is not None and p.fct_callcount > 0):
        print(
            """
Timing Info
-----------
--> <time> <% time> - <total time> <% total time>'

<time>         computation time for this node
<% time>       fraction of total computation time for this node
<total time>   time for this node + total times for this node's ancestors
<% total time> total time for this node over total computation time

N.B.:
* Times include the node time and the function overhead.
* <total time> and <% total time> may over-count computation times
  if inputs to a node share a common ancestor and should be viewed as a
  loose upper bound. Their intended use is to help rule out potential nodes
  to remove when optimizing a graph because their <total time> is very low.
""",
            file=_file,
        )

    op_information = {}

    for r in inputs_to_print:
        _debugprint(
            r,
            prefix="-",
            depth=depth,
            done=done,
            print_type=print_type,
            file=_file,
            ids=ids,
            inner_graph_ops=inner_graph_ops,
            stop_on_name=stop_on_name,
            used_ids=used_ids,
            op_information=op_information,
            parent_node=r.owner,
            print_op_info=print_op_info,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
        )

    for r, p, s, o in zip(outputs_to_print, profile_list, smap, order):

        if hasattr(r.owner, "op"):
            if isinstance(r.owner.op, HasInnerGraph) and r not in inner_graph_ops:
                inner_graph_ops.append(r)
            if print_op_info:
                op_information.update(op_debug_information(r.owner.op, r.owner))

        _debugprint(
            r,
            depth=depth,
            done=done,
            print_type=print_type,
            file=_file,
            order=o,
            ids=ids,
            inner_graph_ops=inner_graph_ops,
            stop_on_name=stop_on_name,
            profile=p,
            smap=s,
            used_ids=used_ids,
            op_information=op_information,
            parent_node=r.owner,
            print_op_info=print_op_info,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
        )

    if len(inner_graph_ops) > 0:
        print("", file=_file)
        new_prefix = " >"
        new_prefix_child = " >"
        print("Inner graphs:", file=_file)

        for s in inner_graph_ops:

            # This is a work-around to maintain backward compatibility
            # (e.g. to only print inner graphs that have been compiled through
            # a call to `Op.prepare_node`)
            inner_fn = getattr(s.owner.op, "_fn", None)

            if inner_fn:
                # If the op was compiled, print the optimized version.
                inner_inputs = inner_fn.maker.fgraph.inputs
                inner_outputs = inner_fn.maker.fgraph.outputs
            else:
                inner_inputs = s.owner.op.inner_inputs
                inner_outputs = s.owner.op.inner_outputs

            outer_inputs = s.owner.inputs

            if hasattr(s.owner.op, "get_oinp_iinp_iout_oout_mappings"):
                inner_to_outer_inputs = {
                    inner_inputs[i]: outer_inputs[o]
                    for i, o in s.owner.op.get_oinp_iinp_iout_oout_mappings()[
                        "outer_inp_from_inner_inp"
                    ].items()
                }
            else:
                inner_to_outer_inputs = None

            if print_op_info:
                op_information.update(op_debug_information(s.owner.op, s.owner))

            print("", file=_file)

            _debugprint(
                s,
                depth=depth,
                done=done,
                print_type=print_type,
                file=_file,
                ids=ids,
                inner_graph_ops=inner_graph_ops,
                stop_on_name=stop_on_name,
                inner_to_outer_inputs=inner_to_outer_inputs,
                used_ids=used_ids,
                op_information=op_information,
                parent_node=s.owner,
                print_op_info=print_op_info,
                print_destroy_map=print_destroy_map,
                print_view_map=print_view_map,
            )

            if print_fgraph_inputs:
                for inp in inner_inputs:
                    _debugprint(
                        r=inp,
                        prefix="-",
                        depth=depth,
                        done=done,
                        print_type=print_type,
                        file=_file,
                        ids=ids,
                        stop_on_name=stop_on_name,
                        inner_graph_ops=inner_graph_ops,
                        inner_to_outer_inputs=inner_to_outer_inputs,
                        used_ids=used_ids,
                        op_information=op_information,
                        parent_node=s.owner,
                        print_op_info=print_op_info,
                        print_destroy_map=print_destroy_map,
                        print_view_map=print_view_map,
                        inner_graph_node=s.owner,
                    )
                inner_to_outer_inputs = None

            for out in inner_outputs:

                if (
                    isinstance(getattr(out.owner, "op", None), HasInnerGraph)
                    and out not in inner_graph_ops
                ):
                    inner_graph_ops.append(out)

                _debugprint(
                    r=out,
                    prefix=new_prefix,
                    depth=depth,
                    done=done,
                    print_type=print_type,
                    file=_file,
                    ids=ids,
                    stop_on_name=stop_on_name,
                    prefix_child=new_prefix_child,
                    inner_graph_ops=inner_graph_ops,
                    inner_to_outer_inputs=inner_to_outer_inputs,
                    used_ids=used_ids,
                    op_information=op_information,
                    parent_node=s.owner,
                    print_op_info=print_op_info,
                    print_destroy_map=print_destroy_map,
                    print_view_map=print_view_map,
                    inner_graph_node=s.owner,
                )

    if file is _file:
        return file
    elif file == "str":
        return _file.getvalue()
    else:
        _file.flush()
    return _file


def _debugprint(
    r: Variable,
    prefix: str = "",
    depth: int = -1,
    done: Optional[Dict[Apply, str]] = None,
    print_type: bool = False,
    file: IOBase = sys.stdout,
    print_destroy_map: bool = False,
    print_view_map: bool = False,
    order: Optional[List[Variable]] = None,
    ids: str = "CHAR",
    stop_on_name: bool = False,
    prefix_child: Optional[str] = None,
    inner_graph_ops: Optional[List[Variable]] = None,
    profile: Optional[ProfileStats] = None,
    inner_to_outer_inputs: Optional[Dict[Variable, Variable]] = None,
    smap: Optional[StorageMapType] = None,
    used_ids: Optional[Dict[Variable, str]] = None,
    op_information: Optional[Dict[Apply, Dict[Variable, str]]] = None,
    parent_node: Optional[Apply] = None,
    print_op_info: bool = False,
    inner_graph_node: Optional[Apply] = None,
) -> IOBase:
    r"""Print the graph leading to `r`.

    Parameters
    ----------
    r
        A `Variable` instance.
    prefix
        Prefix to each line (typically some number of spaces).
    depth
        Print graph to this depth (``-1`` for unlimited).
    done
        A ``dict`` of `Apply` instances that have already been printed and
        their associated printed ids.
        Internal. Used to pass information when recursing.
    print_type
        Whether to print the `Variable`'s type.
    file
        File-like object to which to print.
    print_destroy_map
        Whether to print `Op` ``destroy_map``\s.
    print_view_map
        Whether to print `Op` ``view_map``\s.
    order
        If not empty will print the index in the toposort.
    ids
        Determines the type of identifier used for variables.
          - ``"id"``: print the python id value,
          - ``"int"``: print integer character,
          - ``"CHAR"``: print capital character,
          - ``"auto"``: print the ``auto_name`` value,
          - ``""``: don't print an identifier.
    stop_on_name
        When ``True``, if a node in the graph has a name, we don't print anything
        below it.
    inner_graph_ops
        A list of `Op`\s with inner graphs.
    inner_to_outer_inputs
        A dictionary mapping an `Op`'s inner-inputs to its outer-inputs.
    smap
        ``None`` or the ``storage_map`` when printing an Aesara function.
    used_ids
        A map between nodes and their printed ids.
        It wasn't always printed, but at least a reference to it was printed.
        Internal. Used to pass information when recursing.
    op_information
        Extra `Op`-level information to be added to variable print-outs.
    parent_node
        The parent node of `r`.
    print_op_info
        Print extra information provided by the relevant `Op`\s.  For example,
        print the tap information for `Scan` inputs and outputs.
    inner_graph_node
        The inner-graph node in which `r` is contained.
    """
    if depth == 0:
        return file

    if order is None:
        order = []

    if done is None:
        done = dict()

    if inner_graph_ops is None:
        inner_graph_ops = []

    if print_type:
        type_str = f" <{r.type}>"
    else:
        type_str = ""

    if prefix_child is None:
        prefix_child = prefix

    if used_ids is None:
        used_ids = dict()

    if op_information is None:
        op_information = {}

    def get_id_str(obj, get_printed=True) -> str:
        id_str: str = ""
        if obj in used_ids:
            id_str = used_ids[obj]
        elif obj == "output":
            id_str = "output"
        elif ids == "id":
            id_str = f"[id {id(r)}]"
        elif ids == "int":
            id_str = f"[id {len(used_ids)}]"
        elif ids == "CHAR":
            id_str = f"[id {char_from_number(len(used_ids))}]"
        elif ids == "auto":
            id_str = f"[id {r.auto_name}]"
        elif ids == "":
            id_str = ""
        if get_printed:
            done[obj] = id_str
        used_ids[obj] = id_str
        return id_str

    if hasattr(r.owner, "op"):
        # This variable is the output of a computation, so just print out the
        # `Apply` node
        a = r.owner

        r_name = getattr(r, "name", "")

        if r_name is None:
            r_name = ""
        if r_name:
            r_name = f" '{r_name}'"

        if print_destroy_map and r.owner.op.destroy_map:
            destroy_map_str = f" d={r.owner.op.destroy_map}"
        else:
            destroy_map_str = ""

        if print_view_map and r.owner.op.view_map:
            view_map_str = f" v={r.owner.op.view_map}"
        else:
            view_map_str = ""

        if order:
            o = f" {order.index(r.owner)}"
        else:
            o = ""

        already_done = a in done
        id_str = get_id_str(a)

        if len(a.outputs) == 1:
            idx = ""
        else:
            idx = f".{a.outputs.index(r)}"

        if id_str:
            id_str = f" {id_str}"

        if smap and a.outputs[0] in smap:
            data = f" {smap[a.outputs[0]]}"
        else:
            data = ""

        var_output = f"{prefix}{a.op}{idx}{id_str}{type_str}{r_name}{destroy_map_str}{view_map_str}{o}{data}"

        if print_op_info and r.owner not in op_information:
            op_information.update(op_debug_information(r.owner.op, r.owner))

        node_info = op_information.get(parent_node) or op_information.get(r.owner)
        if node_info and r in node_info:
            var_output = f"{var_output} ({node_info[r]})"

        if profile is None or a not in profile.apply_time:
            print(var_output, file=file)
        else:
            op_time = profile.apply_time[a]
            op_time_percent = (op_time / profile.fct_call_time) * 100
            tot_time_dict = profile.compute_total_times()
            tot_time = tot_time_dict[a]
            tot_time_percent = (tot_time_dict[a] / profile.fct_call_time) * 100

            print(
                "%s --> %8.2es %4.1f%% %8.2es %4.1f%%"
                % (
                    var_output,
                    op_time,
                    op_time_percent,
                    tot_time,
                    tot_time_percent,
                ),
                file=file,
            )

        if not already_done and (
            not stop_on_name or not (hasattr(r, "name") and r.name is not None)
        ):
            new_prefix = prefix_child + " |"
            new_prefix_child = prefix_child + " |"

            for idx, i in enumerate(a.inputs):
                if idx == len(a.inputs) - 1:
                    new_prefix_child = prefix_child + "  "

                if hasattr(i, "owner") and hasattr(i.owner, "op"):
                    if (
                        isinstance(i.owner.op, HasInnerGraph)
                        and i not in inner_graph_ops
                    ):
                        inner_graph_ops.append(i)

                _debugprint(
                    i,
                    new_prefix,
                    depth=depth - 1,
                    done=done,
                    print_type=print_type,
                    file=file,
                    order=order,
                    ids=ids,
                    stop_on_name=stop_on_name,
                    prefix_child=new_prefix_child,
                    inner_graph_ops=inner_graph_ops,
                    profile=profile,
                    inner_to_outer_inputs=inner_to_outer_inputs,
                    smap=smap,
                    used_ids=used_ids,
                    op_information=op_information,
                    parent_node=a,
                    print_op_info=print_op_info,
                    print_destroy_map=print_destroy_map,
                    print_view_map=print_view_map,
                    inner_graph_node=inner_graph_node,
                )
    else:

        id_str = get_id_str(r)

        if id_str:
            id_str = f" {id_str}"

        if smap and r in smap:
            data = f" {smap[r]}"
        else:
            data = ""

        var_output = f"{prefix}{r}{id_str}{type_str}{data}"

        if print_op_info and r.owner and r.owner not in op_information:
            op_information.update(op_debug_information(r.owner.op, r.owner))

        if inner_to_outer_inputs is not None and r in inner_to_outer_inputs:

            outer_r = inner_to_outer_inputs[r]

            if outer_r.owner:
                outer_id_str = get_id_str(outer_r.owner)
            else:
                outer_id_str = get_id_str(outer_r)

            var_output = f"{var_output} -> {outer_id_str}"

        # TODO: This entire approach will only print `Op` info for two levels
        # of nesting.
        for node in dict.fromkeys([inner_graph_node, parent_node, r.owner]):
            node_info = op_information.get(node)
            if node_info and r in node_info:
                var_output = f"{var_output} ({node_info[r]})"

        print(var_output, file=file)

    return file


def _print_fn(op, xin):
    for attr in op.attrs:
        temp = getattr(xin, attr)
        if callable(temp):
            pmsg = temp()
        else:
            pmsg = temp
        print(op.message, attr, "=", pmsg)


class Print(Op):
    """This identity-like Op print as a side effect.

    This identity-like Op has the side effect of printing a message
    followed by its inputs when it runs. Default behaviour is to print
    the __str__ representation. Optionally, one can pass a list of the
    input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print.
                  Functions are identified through callable(), executed and
                  their return value printed.

    :note: WARNING. This can disable some optimizations!
                    (speed and/or stabilization)

            Detailed explanation:
            As of 2012-06-21 the Print op is not known by any optimization.
            Setting a Print op in the middle of a pattern that is usually
            optimized out will block the optimization. for example, log(1+x)
            optimizes to log1p(x) but log(1+Print(x)) is unaffected by
            optimizations.

    """

    view_map = {0: [0]}

    __props__ = ("message", "attrs", "global_fn")

    def __init__(self, message="", attrs=("__str__",), global_fn=_print_fn):
        self.message = message
        self.attrs = tuple(attrs)  # attrs should be a hashable iterable
        self.global_fn = global_fn

    def make_node(self, xin):
        xout = xin.type()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        (xin,) = inputs
        (xout,) = output_storage
        xout[0] = xin
        self.global_fn(self, xin)

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __setstate__(self, dct):
        dct.setdefault("global_fn", _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)


class PrinterState(Scratchpad):
    def __init__(self, props=None, **more_props):
        if props is None:
            props = {}
        elif isinstance(props, Scratchpad):
            self.__update__(props)
        else:
            self.__dict__.update(props)
        self.__dict__.update(more_props)
        # A dict from the object to print to its string
        # representation. If it is a dag and not a tree, it allow to
        # parse each node of the graph only once. They will still be
        # printed many times
        self.memo = {}


class Printer(ABC):
    @abstractmethod
    def process(self, var: Variable, pstate: PrinterState) -> str:
        """Construct a string representation for a `Variable`."""


@contextmanager
def set_precedence(pstate: PrinterState, precedence: int = -1000):
    """Temporarily set the precedence of a `PrinterState`."""
    old_precedence = getattr(pstate, "precedence", None)
    pstate.precedence = precedence
    try:
        yield
    finally:
        pstate.precedence = old_precedence


class OperatorPrinter(Printer):
    def __init__(self, operator, precedence, assoc="left"):
        self.operator = operator
        self.precedence = precedence
        self.assoc = assoc
        assert self.assoc in VALID_ASSOC

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"operator {self.operator} cannot represent a variable that is "
                "not the result of an operation"
            )

        # Precedence seems to be buggy, see #249
        # So, in doubt, we parenthesize everything.
        # outer_precedence = getattr(pstate, 'precedence', -999999)
        # outer_assoc = getattr(pstate, 'assoc', 'none')
        # if outer_precedence > self.precedence:
        #    parenthesize = True
        # else:
        #    parenthesize = False
        parenthesize = True

        input_strings = []
        max_i = len(node.inputs) - 1
        for i, input in enumerate(node.inputs):
            new_precedence = self.precedence
            if self.assoc == "left" and i != 0 or self.assoc == "right" and i != max_i:
                new_precedence += 1e-6

            with set_precedence(pstate, new_precedence):
                s = pprinter.process(input, pstate)

            input_strings.append(s)
        if len(input_strings) == 1:
            s = self.operator + input_strings[0]
        else:
            s = f" {self.operator} ".join(input_strings)
        if parenthesize:
            r = f"({s})"
        else:
            r = s
        pstate.memo[output] = r
        return r


class PatternPrinter(Printer):
    def __init__(self, *patterns):
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, str):
                self.patterns.append((pattern, ()))
            else:
                self.patterns.append((pattern[0], pattern[1:]))

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"Patterns {self.patterns} cannot represent a variable that is "
                "not the result of an operation"
            )
        idx = node.outputs.index(output)
        pattern, precedences = self.patterns[idx]
        precedences += (1000,) * len(node.inputs)

        def pp_process(input, new_precedence):
            with set_precedence(pstate, new_precedence):
                r = pprinter.process(input, pstate)
            return r

        d = {
            str(i): x
            for i, x in enumerate(
                pp_process(input, precedence)
                for input, precedence in zip(node.inputs, precedences)
            )
        }
        r = pattern % d
        pstate.memo[output] = r
        return r


class FunctionPrinter(Printer):
    def __init__(self, names: List[str], keywords: Optional[List[str]] = None):
        """
        Parameters
        ----------
        names
            The function names used for each output.
        keywords
            The `Op` keywords to include in the output.
        """
        self.names = names

        if keywords is None:
            keywords = []

        self.keywords = keywords

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"function {self.names} cannot represent a variable that is "
                "not the result of an operation"
            )
        idx = node.outputs.index(output)
        name = self.names[idx]
        with set_precedence(pstate):
            inputs_str = ", ".join(
                [pprinter.process(input, pstate) for input in node.inputs]
            )
            keywords_str = ", ".join(
                [f"{kw}={getattr(node.op, kw)}" for kw in self.keywords]
            )

            if keywords_str and inputs_str:
                keywords_str = f", {keywords_str}"

            r = f"{name}({inputs_str}{keywords_str})"

        pstate.memo[output] = r
        return r


class IgnorePrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"function {self.function} cannot represent a variable that is"
                " not the result of an operation"
            )
        input = node.inputs[0]
        r = f"{pprinter.process(input, pstate)}"
        pstate.memo[output] = r
        return r


class LeafPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        if output.name in greek:
            r = greek[output.name]
        else:
            r = str(output)
        pstate.memo[output] = r
        return r


leaf_printer = LeafPrinter()


class ConstantPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        r = str(output.data)
        pstate.memo[output] = r
        return r


constant_printer = ConstantPrinter()


class DefaultPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            return leaf_printer.process(output, pstate)
        with set_precedence(pstate):
            r = "{}({})".format(
                str(node.op),
                ", ".join([pprinter.process(input, pstate) for input in node.inputs]),
            )

        pstate.memo[output] = r
        return r


default_printer = DefaultPrinter()


class PPrinter(Printer):
    def __init__(self):
        self.printers: List[Tuple[Union[Op, type, Callable], Printer]] = []
        self.printers_dict: Dict[Union[Op, type, Callable], Printer] = {}

    def assign(self, condition: Union[Op, type, Callable], printer: Printer):
        if isinstance(condition, (Op, type)):
            self.printers_dict[condition] = printer
        else:
            self.printers.insert(0, (condition, printer))

    def process(self, r: Variable, pstate: Optional[PrinterState] = None) -> str:
        if pstate is None:
            pstate = PrinterState(pprinter=self)
        elif isinstance(pstate, dict):
            pstate = PrinterState(pprinter=self, **pstate)
        if getattr(r, "owner", None) is not None:
            if r.owner.op in self.printers_dict:
                return self.printers_dict[r.owner.op].process(r, pstate)
            if type(r.owner.op) in self.printers_dict:
                return self.printers_dict[type(r.owner.op)].process(r, pstate)
        for condition, printer in self.printers:
            if condition(pstate, r):
                return printer.process(r, pstate)
        return ""

    def clone(self):
        cp = copy(self)
        cp.printers = list(self.printers)
        cp.printers_dict = dict(self.printers_dict)
        return cp

    def clone_assign(self, condition, printer):
        cp = self.clone()
        cp.assign(condition, printer)
        return cp

    def process_graph(self, inputs, outputs, updates=None, display_inputs=False):
        if updates is None:
            updates = {}
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        current = None
        if display_inputs:
            strings = [
                (0, "inputs: " + ", ".join(map(str, list(inputs) + updates.keys())))
            ]
        else:
            strings = []
        pprinter = self.clone_assign(
            lambda pstate, r: r.name is not None and r is not current, leaf_printer
        )
        inv_updates = {b: a for (a, b) in updates.items()}
        i = 1
        for node in io_toposort(
            list(inputs) + updates.keys(), list(outputs) + updates.values()
        ):
            for output in node.outputs:
                if output in inv_updates:
                    name = str(inv_updates[output])
                    strings.append((i + 1000, f"{name} <- {pprinter.process(output)}"))
                    i += 1
                if output.name is not None or output in outputs:
                    if output.name is None:
                        name = "out[%i]" % outputs.index(output)
                    else:
                        name = output.name
                    # backport
                    # name = 'out[%i]' % outputs.index(output) if output.name
                    #  is None else output.name
                    current = output
                    try:
                        idx = 2000 + outputs.index(output)
                    except ValueError:
                        idx = i
                    if len(outputs) == 1 and outputs[0] is output:
                        strings.append((idx, f"return {pprinter.process(output)}"))
                    else:
                        strings.append((idx, f"{name} = {pprinter.process(output)}"))
                    i += 1
        strings.sort()
        return "\n".join(s[1] for s in strings)

    def __call__(self, *args):
        if len(args) == 1:
            return self.process(*args)
        elif len(args) == 2 and isinstance(args[1], (PrinterState, dict)):
            return self.process(*args)
        elif len(args) > 2:
            return self.process_graph(*args)
        else:
            raise TypeError("Not enough arguments to call.")


use_ascii = True

if use_ascii:
    special = dict(middle_dot="\\dot", big_sigma="\\Sigma")

    greek = dict(
        alpha="\\alpha",
        beta="\\beta",
        gamma="\\gamma",
        delta="\\delta",
        epsilon="\\epsilon",
    )
else:

    special = dict(middle_dot="\u00B7", big_sigma="\u03A3")

    greek = dict(
        alpha="\u03B1",
        beta="\u03B2",
        gamma="\u03B3",
        delta="\u03B4",
        epsilon="\u03B5",
    )


pprint: PPrinter = PPrinter()
pprint.assign(lambda pstate, r: True, default_printer)
pprint.assign(lambda pstate, r: isinstance(r, Constant), constant_printer)


pp = pprint
"""
Print to the terminal a math-like expression.
"""

# colors not used: orange, amber#FFBF00, purple, pink,
# used by default: green, blue, grey, red
default_colorCodes = {
    "Scan": "yellow",
    "Shape": "brown",
    "IfElse": "magenta",
    "Elemwise": "#FFAABB",  # dark pink
    "Subtensor": "#FFAAFF",  # purple
    "Alloc": "#FFAA22",  # orange
    "Output": "blue",
}


def pydotprint(
    fct,
    outfile=None,
    compact=True,
    format="png",
    with_ids=False,
    high_contrast=True,
    cond_highlight=None,
    colorCodes=None,
    max_label_size=70,
    scan_graphs=False,
    var_with_name_simple=False,
    print_output_file=True,
    return_image=False,
):
    """Print to a file the graph of a compiled aesara function's ops. Supports
    all pydot output formats, including png and svg.

    :param fct: a compiled Aesara function, a Variable, an Apply or
                a list of Variable.
    :param outfile: the output file where to put the graph.
    :param compact: if True, will remove intermediate var that don't have name.
    :param format: the file format of the output.
    :param with_ids: Print the toposort index of the node in the node name.
                     and an index number in the variable ellipse.
    :param high_contrast: if true, the color that describes the respective
            node is filled with its corresponding color, instead of coloring
            the border
    :param colorCodes: dictionary with names of ops as keys and colors as
            values
    :param cond_highlight: Highlights a lazy if by surrounding each of the 3
                possible categories of ops with a border. The categories
                are: ops that are on the left branch, ops that are on the
                right branch, ops that are on both branches
                As an alternative you can provide the node that represents
                the lazy if
    :param scan_graphs: if true it will plot the inner graph of each scan op
                in files with the same name as the name given for the main
                file to which the name of the scan op is concatenated and
                the index in the toposort of the scan.
                This index can be printed with the option with_ids.
    :param var_with_name_simple: If true and a variable have a name,
                we will print only the variable name.
                Otherwise, we concatenate the type to the var name.
    :param return_image: If True, it will create the image and return it.
        Useful to display the image in ipython notebook.

        .. code-block:: python

            import aesara
            v = aesara.tensor.vector()
            from IPython.display import SVG
            SVG(aesara.printing.pydotprint(v*2, return_image=True,
                                           format='svg'))

    In the graph, ellipses are Apply Nodes (the execution of an op)
    and boxes are variables.  If variables have names they are used as
    text (if multiple vars have the same name, they will be merged in
    the graph).  Otherwise, if the variable is constant, we print its
    value and finally we print the type + a unique number to prevent
    multiple vars from being merged.  We print the op of the apply in
    the Apply box with a number that represents the toposort order of
    application of those Apply.  If an Apply has more than 1 input, we
    label each edge between an input and the Apply node with the
    input's index.

    Variable color code::
        - Cyan boxes are SharedVariable, inputs and/or outputs) of the graph,
        - Green boxes are inputs variables to the graph,
        - Blue boxes are outputs variables of the graph,
        - Grey boxes are variables that are not outputs and are not used,

    Default apply node code::
        - Red ellipses are transfers from/to the gpu
        - Yellow are scan node
        - Brown are shape node
        - Magenta are IfElse node
        - Dark pink are elemwise node
        - Purple are subtensor
        - Orange are alloc node

    For edges, they are black by default. If a node returns a view
    of an input, we put the corresponding input edge in blue. If it
    returns a destroyed input, we put the corresponding edge in red.

    .. note::

        Since October 20th, 2014, this print the inner function of all
        scan separately after the top level debugprint output.

    """
    from aesara.scan.op import Scan

    if colorCodes is None:
        colorCodes = default_colorCodes

    if outfile is None:
        outfile = os.path.join(
            config.compiledir, "aesara.pydotprint." + config.device + "." + format
        )

    if isinstance(fct, Function):
        profile = getattr(fct, "profile", None)
        fgraph = fct.maker.fgraph
        outputs = fgraph.outputs
        topo = fgraph.toposort()
    elif isinstance(fct, FunctionGraph):
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
        fgraph = fct
    else:
        if isinstance(fct, Variable):
            fct = [fct]
        elif isinstance(fct, Apply):
            fct = fct.outputs
        assert isinstance(fct, (list, tuple))
        assert all(isinstance(v, Variable) for v in fct)
        fct = FunctionGraph(inputs=list(graph_inputs(fct)), outputs=fct)
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
        fgraph = fct
    if not pydot_imported:
        raise RuntimeError(
            "Failed to import pydot. You must install graphviz "
            "and either pydot or pydot-ng for "
            f"`pydotprint` to work:\n {pydot_imported_msg}",
        )

    g = pd.Dot()

    if cond_highlight is not None:
        c1 = pd.Cluster("Left")
        c2 = pd.Cluster("Right")
        c3 = pd.Cluster("Middle")
        cond = None
        for node in topo:
            if (
                node.op.__class__.__name__ == "IfElse"
                and node.op.name == cond_highlight
            ):
                cond = node
        if cond is None:
            _logger.warning(
                "pydotprint: cond_highlight is set but there is no"
                " IfElse node in the graph"
            )
            cond_highlight = None

    if cond_highlight is not None:

        def recursive_pass(x, ls):
            if not x.owner:
                return ls
            else:
                ls += [x.owner]
                for inp in x.inputs:
                    ls += recursive_pass(inp, ls)
                return ls

        left = set(recursive_pass(cond.inputs[1], []))
        right = set(recursive_pass(cond.inputs[2], []))
        middle = left.intersection(right)
        left = left.difference(middle)
        right = right.difference(middle)
        middle = list(middle)
        left = list(left)
        right = list(right)

    var_str = {}
    var_id = {}
    all_strings = set()

    def var_name(var):
        if var in var_str:
            return var_str[var], var_id[var]

        if var.name is not None:
            if var_with_name_simple:
                varstr = var.name
            else:
                varstr = "name=" + var.name + " " + str(var.type)
        elif isinstance(var, Constant):
            dstr = "val=" + str(np.asarray(var.data))
            if "\n" in dstr:
                dstr = dstr[: dstr.index("\n")]
            varstr = f"{dstr} {var.type}"
        elif var in input_update and input_update[var].name is not None:
            varstr = input_update[var].name
            if not var_with_name_simple:
                varstr += str(var.type)
        else:
            # a var id is needed as otherwise var with the same type will be
            # merged in the graph.
            varstr = str(var.type)
        if len(varstr) > max_label_size:
            varstr = varstr[: max_label_size - 3] + "..."
        var_str[var] = varstr
        var_id[var] = str(id(var))

        all_strings.add(varstr)

        return varstr, var_id[var]

    apply_name_cache = {}
    apply_name_id = {}

    def apply_name(node):
        if node in apply_name_cache:
            return apply_name_cache[node], apply_name_id[node]
        prof_str = ""
        if profile:
            time = profile.apply_time.get((fgraph, node), 0)
            # second, %fct time in profiler
            if profile.fct_callcount == 0 or profile.fct_call_time == 0:
                pf = 0
            else:
                pf = time * 100 / profile.fct_call_time
            prof_str = f"   ({time:.3f}s,{pf:.3f}%)"
        applystr = str(node.op).replace(":", "_")
        applystr += prof_str
        if (applystr in all_strings) or with_ids:
            idx = " id=" + str(topo.index(node))
            if len(applystr) + len(idx) > max_label_size:
                applystr = applystr[: max_label_size - 3 - len(idx)] + idx + "..."
            else:
                applystr = applystr + idx
        elif len(applystr) > max_label_size:
            applystr = applystr[: max_label_size - 3] + "..."
            idx = 1
            while applystr in all_strings:
                idx += 1
                suffix = " id=" + str(idx)
                applystr = applystr[: max_label_size - 3 - len(suffix)] + "..." + suffix

        all_strings.add(applystr)
        apply_name_cache[node] = applystr
        apply_name_id[node] = str(id(node))

        return applystr, apply_name_id[node]

    # Update the inputs that have an update function
    input_update = {}
    reverse_input_update = {}
    # Here outputs can be the original list, as we should not change
    # it, we must copy it.
    outputs = list(outputs)
    if isinstance(fct, Function):

        # TODO: Get rid of all this `expanded_inputs` nonsense and use
        # `fgraph.update_mapping`
        function_inputs = zip(fct.maker.expanded_inputs, fgraph.inputs)
        for i, fg_ii in reversed(list(function_inputs)):
            if i.update is not None:
                k = outputs.pop()
                # Use the fgaph.inputs as it isn't the same as maker.inputs
                input_update[k] = fg_ii
                reverse_input_update[fg_ii] = k

    apply_shape = "ellipse"
    var_shape = "box"
    for node_idx, node in enumerate(topo):
        astr, aid = apply_name(node)

        use_color = None
        for opName, color in colorCodes.items():
            if opName in node.op.__class__.__name__:
                use_color = color

        if use_color is None:
            nw_node = pd.Node(aid, label=astr, shape=apply_shape)
        elif high_contrast:
            nw_node = pd.Node(
                aid, label=astr, style="filled", fillcolor=use_color, shape=apply_shape
            )
        else:
            nw_node = pd.Node(aid, label=astr, color=use_color, shape=apply_shape)
        g.add_node(nw_node)
        if cond_highlight:
            if node in middle:
                c3.add_node(nw_node)
            elif node in left:
                c1.add_node(nw_node)
            elif node in right:
                c2.add_node(nw_node)

        for idx, var in enumerate(node.inputs):
            varstr, varid = var_name(var)
            label = ""
            if len(node.inputs) > 1:
                label = str(idx)
            param = {}
            if label:
                param["label"] = label
            if node.op.view_map and idx in reduce(
                list.__add__, node.op.view_map.values(), []
            ):
                param["color"] = colorCodes["Output"]
            elif node.op.destroy_map and idx in reduce(
                list.__add__, node.op.destroy_map.values(), []
            ):
                param["color"] = "red"
            if var.owner is None:
                color = "green"
                if isinstance(var, SharedVariable):
                    # Input are green, output blue
                    # Mixing blue and green give cyan! (input and output var)
                    color = "cyan"
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            fillcolor=color,
                            label=varstr,
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(varid, color=color, label=varstr, shape=var_shape)
                    )
                g.add_edge(pd.Edge(varid, aid, **param))
            elif var.name or not compact or var in outputs:
                g.add_edge(pd.Edge(varid, aid, **param))
            else:
                # no name, so we don't make a var ellipse
                if label:
                    label += " "
                label += str(var.type)
                if len(label) > max_label_size:
                    label = label[: max_label_size - 3] + "..."
                param["label"] = label
                g.add_edge(pd.Edge(apply_name(var.owner)[1], aid, **param))

        for idx, var in enumerate(node.outputs):
            varstr, varid = var_name(var)
            out = var in outputs
            label = ""
            if len(node.outputs) > 1:
                label = str(idx)
            if len(label) > max_label_size:
                label = label[: max_label_size - 3] + "..."
            param = {}
            if label:
                param["label"] = label
            if out or var in input_update:
                g.add_edge(pd.Edge(aid, varid, **param))
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            label=varstr,
                            fillcolor=colorCodes["Output"],
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(
                            varid,
                            color=colorCodes["Output"],
                            label=varstr,
                            shape=var_shape,
                        )
                    )
            elif len(fgraph.clients[var]) == 0:
                g.add_edge(pd.Edge(aid, varid, **param))
                # grey mean that output var isn't used
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            label=varstr,
                            fillcolor="grey",
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(varid, label=varstr, color="grey", shape=var_shape)
                    )
            elif var.name or not compact:
                if not (not compact):
                    if label:
                        label += " "
                    label += str(var.type)
                    if len(label) > max_label_size:
                        label = label[: max_label_size - 3] + "..."
                    param["label"] = label
                g.add_edge(pd.Edge(aid, varid, **param))
                g.add_node(pd.Node(varid, shape=var_shape, label=varstr))
    #            else:
    # don't add edge here as it is already added from the inputs.

    # The var that represent updates, must be linked to the input var.
    for sha, up in input_update.items():
        _, shaid = var_name(sha)
        _, upid = var_name(up)
        g.add_edge(pd.Edge(shaid, upid, label="UPDATE", color=colorCodes["Output"]))

    if cond_highlight:
        g.add_subgraph(c1)
        g.add_subgraph(c2)
        g.add_subgraph(c3)

    if not outfile.endswith("." + format):
        outfile += "." + format

    if scan_graphs:
        scan_ops = [(idx, x) for idx, x in enumerate(topo) if isinstance(x.op, Scan)]
        path, fn = os.path.split(outfile)
        basename = ".".join(fn.split(".")[:-1])
        # Safe way of doing things .. a file name may contain multiple .
        ext = fn[len(basename) :]

        for idx, scan_op in scan_ops:
            # is there a chance that name is not defined?
            if hasattr(scan_op.op, "name"):
                new_name = basename + "_" + scan_op.op.name + "_" + str(idx)
            else:
                new_name = basename + "_" + str(idx)
            new_name = os.path.join(path, new_name + ext)
            if hasattr(scan_op.op, "_fn"):
                to_print = scan_op.op.fn
            else:
                to_print = scan_op.op.inner_outputs
            pydotprint(
                to_print,
                new_name,
                compact,
                format,
                with_ids,
                high_contrast,
                cond_highlight,
                colorCodes,
                max_label_size,
                scan_graphs,
            )

    if return_image:
        return g.create(prog="dot", format=format)
    else:
        try:
            g.write(outfile, prog="dot", format=format)
        except pd.InvocationException:
            # based on https://github.com/Theano/Theano/issues/2988
            version = getattr(pd, "__version__", "")
            if version and [int(n) for n in version.split(".")] < [1, 0, 28]:
                raise Exception(
                    "Old version of pydot detected, which can "
                    "cause issues with pydot printing. Try "
                    "upgrading pydot version to a newer one"
                )
            raise

        if print_output_file:
            print("The output file is available at", outfile)


class _TagGenerator:
    """Class for giving abbreviated tags like to objects.
    Only really intended for internal use in order to
    implement min_informative_st"""

    def __init__(self):
        self.cur_tag_number = 0

    def get_tag(self):
        rval = char_from_number(self.cur_tag_number)

        self.cur_tag_number += 1

        return rval


def min_informative_str(obj, indent_level=0, _prev_obs=None, _tag_generator=None):
    """
    Returns a string specifying to the user what obj is
    The string will print out as much of the graph as is needed
    for the whole thing to be specified in terms only of constants
    or named variables.


    Parameters
    ----------
    obj: the name to convert to a string
    indent_level: the number of tabs the tree should start printing at
                  (nested levels of the tree will get more tabs)
    _prev_obs: should only be used by min_informative_str
                    a dictionary mapping previously converted
                    objects to short tags


    Basic design philosophy
    -----------------------

    The idea behind this function is that it can be used as parts of
    command line tools for debugging or for error messages. The
    information displayed is intended to be concise and easily read by
    a human. In particular, it is intended to be informative when
    working with large graphs composed of subgraphs from several
    different people's code, as in pylearn2.

    Stopping expanding subtrees when named variables are encountered
    makes it easier to understand what is happening when a graph
    formed by composing several different graphs made by code written
    by different authors has a bug.

    An example output is:

    A. Elemwise{add_no_inplace}
        B. log_likelihood_v_given_h
        C. log_likelihood_h


    If the user is told they have a problem computing this value, it's
    obvious that either log_likelihood_h or log_likelihood_v_given_h
    has the wrong dimensionality. The variable's str object would only
    tell you that there was a problem with an
    Elemwise{add_no_inplace}. Since there are many such ops in a
    typical graph, such an error message is considerably less
    informative. Error messages based on this function should convey
    much more information about the location in the graph of the error
    while remaining succinct.

    One final note: the use of capital letters to uniquely identify
    nodes within the graph is motivated by legibility. I do not use
    numbers or lower case letters since these are pretty common as
    parts of names of ops, etc. I also don't use the object's id like
    in debugprint because it gives such a long string that takes time
    to visually diff.

    """

    if _prev_obs is None:
        _prev_obs = {}

    indent = " " * indent_level

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return indent + "<" + tag + ">"

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, "__array__"):
        name = "<ndarray>"
    elif hasattr(obj, "name") and obj.name is not None:
        name = obj.name
    elif hasattr(obj, "owner") and obj.owner is not None:
        name = str(obj.owner.op)
        for ipt in obj.owner.inputs:
            name += "\n"
            name += min_informative_str(
                ipt,
                indent_level=indent_level + 1,
                _prev_obs=_prev_obs,
                _tag_generator=_tag_generator,
            )
    else:
        name = str(obj)

    prefix = cur_tag + ". "

    rval = indent + prefix + name

    return rval


def var_descriptor(obj, _prev_obs=None, _tag_generator=None):
    """
    Returns a string, with no endlines, fully specifying
    how a variable is computed. Does not include any memory
    location dependent information such as the id of a node.
    """
    if _prev_obs is None:
        _prev_obs = {}

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return "<" + tag + ">"

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, "__array__"):
        # hashlib hashes only the contents of the buffer, but
        # it can have different semantics depending on the strides
        # of the ndarray
        name = "<ndarray:"
        name += "strides=[" + ",".join(str(stride) for stride in obj.strides) + "]"
        name += ",digest=" + hashlib.sha256(obj).hexdigest() + ">"
    elif hasattr(obj, "owner") and obj.owner is not None:
        name = str(obj.owner.op) + "("
        name += ",".join(
            var_descriptor(ipt, _prev_obs=_prev_obs, _tag_generator=_tag_generator)
            for ipt in obj.owner.inputs
        )
        name += ")"
    elif hasattr(obj, "name") and obj.name is not None:
        # Only print the name if there is no owner.
        # This way adding a name to an intermediate node can't make
        # a deeper graph get the same descriptor as a shallower one
        name = obj.name
    else:
        name = str(obj)
        if " at 0x" in name:
            # The __str__ method is encoding the object's id in its str
            name = position_independent_str(obj)
            if " at 0x" in name:
                print(name)
                raise AssertionError()

    prefix = cur_tag + "="

    rval = prefix + name

    return rval


def position_independent_str(obj):
    if isinstance(obj, Variable):
        rval = "aesara_var"
        rval += "{type=" + str(obj.type) + "}"
    else:
        raise NotImplementedError()

    return rval


def hex_digest(x):
    """
    Returns a short, mostly hexadecimal hash of a numpy ndarray
    """
    assert isinstance(x, np.ndarray)
    rval = hashlib.sha256(x.tostring()).hexdigest()
    # hex digest must be annotated with strides to avoid collisions
    # because the buffer interface only exposes the raw data, not
    # any info about the semantics of how that data should be arranged
    # into a tensor
    rval = rval + "|strides=[" + ",".join(str(stride) for stride in x.strides) + "]"
    rval = rval + "|shape=[" + ",".join(str(s) for s in x.shape) + "]"
    return rval


def get_node_by_id(
    graphs: Iterable[Variable], target_var_id: str, ids: str = "CHAR"
) -> Optional[Union[Variable, Apply]]:
    r"""Get `Apply` nodes or `Variable`\s in a graph using their `debugprint` IDs.

    Parameters
    ----------
    graphs:
        The graph, or graphs, to search.
    target_var_id:
        The name to search for.
    ids:
        The ID scheme to use (see `debugprint.`).

    Returns
    -------
    The `Apply`/`Variable` matching `target_var_id` or ``None``.

    """
    from aesara.printing import debugprint

    if isinstance(graphs, Variable):
        graphs = (graphs,)

    used_ids: Dict[Variable, str] = {}

    _ = debugprint(graphs, file="str", used_ids=used_ids, ids=ids)

    id_to_node = {v: k for k, v in used_ids.items()}

    id_str = f"[id {target_var_id}]"

    return id_to_node.get(id_str, None)

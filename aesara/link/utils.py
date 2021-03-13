import io
import sys
import traceback
import warnings
from operator import itemgetter
from typing import Callable, Dict, Iterable, List, NoReturn, Optional, Tuple, Union

import numpy as np

from aesara import utils
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant
from aesara.graph.fg import FunctionGraph


def map_storage(
    fgraph: FunctionGraph,
    order: Iterable[Apply],
    input_storage: Optional[List],
    output_storage: Optional[List],
    storage_map: Dict = None,
) -> Tuple[List, List, Dict]:
    """Ensure there is storage (a length-1 list) for inputs, outputs, and interior nodes.

    :param fgraph: The current fgraph.  This function uses the inputs and outputs attributes.
    :param order: an iterable over Apply instances (in program running order)
    :param input_storage: None or existing input storage (see below)
    :param output_storage: None or existing output storage (see below)

    :rtype: 3-tuple
    :returns: (list of storage for inputs, list of storage for outputs, and the `storage_map`)

    Parameters
    ----------
    fgraph
        The current fgraph. This function uses the inputs and outputs
        attributes.
    order
        An iterable over Apply instances (in program running order).
    input_storage
        None or existing input storage (see below).
    output_storage
        None or existing output storage (see below).

    Returns
    -------
    3-tuple
        List of storage for inputs, list of storage for outputs, and
        the `storage_map`.

    Extended summary
    ----------------
    This function iterates over the nodes in `order` and ensures that for every
    input and output `Variable`, there is a unique storage container. This is
    returned as a dictionary Variable -> storage called the `storage_map`.

    This function also returns `input_storage`, which is a list of storages
    corresponding to fgraph.inputs.
    This function also returns `output_storage`, which is a list of storages
    corresponding to fgraph.outputs.

    """
    # each Apply argument's data is stored in a list of length 1 (these lists act like pointers)

    if storage_map is None:
        storage_map = {}

    # input_storage is a list of data-containers for the inputs.
    if input_storage is None:
        input_storage = [[None] for input in fgraph.inputs]
    else:
        assert len(fgraph.inputs) == len(input_storage)

    # add input storage into storage_map
    for r, storage in zip(fgraph.inputs, input_storage):
        if r in storage_map:
            assert storage_map[r] is storage, (
                "Given input_storage conflicts "
                "with storage in given storage_"
                "map. Given input_storage: ",
                storage,
                "Storage in storage_ma" "p: ",
                storage_map[r],
            )
        else:
            storage_map[r] = storage
    #     for orphan in fgraph.orphans:
    #         if not isinstance(orphan, Constant):
    #             raise TypeError("Cannot link a graph with non-constant orphans.", orphan)
    #         storage_map[orphan] = [orphan.data]

    # allocate output storage
    if output_storage is not None:
        assert len(fgraph.outputs) == len(output_storage)
        for r, storage in zip(fgraph.outputs, output_storage):
            if r in storage_map:
                assert storage_map[r] is storage, (
                    "Given output_storage confl"
                    "icts with storage in given"
                    " storage_map. Given output"
                    "_storage: ",
                    storage,
                    "Sto" "rage in storage_map: ",
                    storage_map[r],
                )
            else:
                storage_map[r] = storage

    # allocate storage for intermediate computation
    for node in order:
        for r in node.inputs:
            if r not in storage_map:
                assert isinstance(r, Constant)
                storage_map[r] = [r.data]
        for r in node.outputs:
            storage_map.setdefault(r, [None])
    for r in fgraph.outputs:
        if isinstance(r, Constant):
            storage_map.setdefault(r, [r.data])

    # extract output storage
    if output_storage is None:
        output_storage = [storage_map[r] for r in fgraph.outputs]

    return input_storage, output_storage, storage_map


def streamline(
    fgraph: FunctionGraph,
    thunks,
    order,
    post_thunk_old_storage=None,
    no_recycling=None,
    nice_errors=True,
) -> Callable[[], NoReturn]:
    """
    WRITEME

    Parameters
    ----------
    fgraph
    thunks
        The list of program instructions.
    order
        The list of apply instances that gave rise to the thunks
        (same order as thunks).
    post_thunk_old_storage
        A list (corresponding to thunks, order) whose elements are lists of
        storage cells, that should be cleared after running thecorresponding
        thunk. A value of None disables this functionality.
    no_recycling
        Storage elements that cannot be 'recycled' by repeatedly executing the
        program. These storage elements are cleared before re-running.
    nice_errors
        Run in such a way that the double-traceback is printed. This costs a
        bit of performance in the inner python loop.

    """
    if no_recycling is None:
        no_recycling = []

    if len(thunks) != len(order):
        raise ValueError(
            "Length of thunks and order must match", (len(thunks), len(order))
        )

    if post_thunk_old_storage:
        if len(thunks) != len(post_thunk_old_storage):
            raise ValueError(
                "Length of thunks and post_thunk_old_storage must match",
                (len(thunks), len(post_thunk_old_storage)),
            )

        def streamline_default_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node, old_storage in zip(
                    thunks, order, post_thunk_old_storage
                ):
                    thunk()
                    for old_s in old_storage:
                        old_s[0] = None
            except Exception:
                raise_with_op(fgraph, node, thunk)

        f = streamline_default_f
    elif nice_errors:

        def streamline_nice_errors_f():
            for x in no_recycling:
                x[0] = None
            try:
                for thunk, node in zip(thunks, order):
                    thunk()
            except Exception:
                raise_with_op(fgraph, node, thunk)

        f = streamline_nice_errors_f
    else:
        # don't worry about raise_with_op, just go a little faster.
        # there is a mix of python and c thunks
        def streamline_fast_f():
            for x in no_recycling:
                x[0] = None
            for thunk in thunks:
                thunk()

        f = streamline_fast_f
    return f


def gc_helper(node_list: List[Apply]):
    """
    Return the set of Variable instances which are computed by node_list.
    Parameters
    ----------
    node_list
        List of Apply instances in program execution order.

    Returns
    -------
    2-tuple
        FIRST, the set of Variable instances which are computed by node_list,
        and SECOND a dictionary that maps each Variable instance to a the last
        node to use Variable as an input.

    Extended Summary
    ----------------
    This is used to allow garbage collection within graphs.

    It ignores view_map and destroy_map. This isn't needed as python
    have reference count. In Aesara gc, we should not take into
    account view_map and destroy_map as if the thunk decided to create
    a new output, we would delay uselessly its gc by Python.

    """
    # for freeing memory
    last_user = {}
    computed = set()
    for node in node_list:
        for input in node.inputs:
            last_user[input] = node
        for output in node.outputs:
            computed.add(output)
    return computed, last_user


def raise_with_op(
    fgraph: FunctionGraph, node, thunk=None, exc_info=None, storage_map=None
):
    """
    Re-raise an exception while annotating the exception object with
    debug info.

    Parameters
    ----------
    node : Apply node
        The Apply node object that resulted in the raised exception.
    exc_info : tuple, optional
        A tuple containing the exception type, exception object and
        associated traceback, as would be returned by a call to
        `sys.exc_info()` (which is done if `None` is passed).
    storage_map: dict, optional
        storage map of the aesara function that resulted in the
        raised exception.

    Notes
    -----
    This re-raises the exception described by `exc_info` (or the last
    one raised, if `exc_info` is omitted) and annotates the exception
    object with several new members which may be helpful for debugging
    Aesara graphs. They are:

     * __op_instance__: The Op that is responsible for the exception
       being raised.
     * __thunk_trace__: A traceback corresponding to the code that
       actually generated the exception, if it is available.
     * __applynode_index__: The index of the Apply node corresponding
       to this op in `op.fgraph.toposort()`.

    The exception is not annotated if it is of type `KeyboardInterrupt`.

    TODO: Make this work with linker defined schedule
    """
    verbosity = config.exception_verbosity

    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_trace = exc_info
    if exc_type == KeyboardInterrupt:
        # print a simple traceback from KeyboardInterrupt
        raise exc_value.with_traceback(exc_trace)
    try:
        trace = node.outputs[0].tag.trace
    except AttributeError:
        try:
            trace = node.op.tag.trace
        except AttributeError:
            trace = ()
    exc_value.__thunk_trace__ = trace
    exc_value.__op_instance__ = node
    topo = fgraph.toposort()
    if node in topo:
        node_index = topo.index(node)
    else:
        node_index = None
    exc_value.__applynode_index__ = node_index

    hints = []
    detailed_err_msg = "\nApply node that caused the error: " + str(node)
    if exc_value.__applynode_index__ is not None:
        detailed_err_msg += f"\nToposort index: {int(node_index)}"

    types = [getattr(ipt, "type", "No type") for ipt in node.inputs]
    detailed_err_msg += f"\nInputs types: {types}\n"

    shapes: Union[List, str]
    strides: Union[List, str]
    scalar_values: Union[List, str]

    if thunk is not None:
        if hasattr(thunk, "inputs"):
            shapes = [getattr(ipt[0], "shape", "No shapes") for ipt in thunk.inputs]
            strides = [getattr(ipt[0], "strides", "No strides") for ipt in thunk.inputs]
            scalar_values = []
            for ipt in thunk.inputs:
                if getattr(ipt[0], "size", -1) <= 5:
                    scalar_values.append(ipt[0])
                else:
                    scalar_values.append("not shown")
        else:
            shapes = "The thunk doesn't have an `inputs` attributes."
            strides = "So we can't access the strides of the input values"
            scalar_values = "and we can't print its scalar input values"
        clients = [[c[0] for c in fgraph.clients[var]] for var in node.outputs]
        detailed_err_msg += (
            f"Inputs shapes: {shapes}"
            + f"\nInputs strides: {strides}"
            + f"\nInputs values: {scalar_values}"
        )
        if verbosity == "high":
            detailed_err_msg += "\nInputs type_num: %s" % str(
                [getattr(getattr(i[0], "dtype", ""), "num", "") for i in thunk.inputs]
            )
        if hasattr(node.op, "__input_name__"):
            detailed_err_msg += f"\nInputs name: {node.op.__input_name__}\n"

        detailed_err_msg += f"\nOutputs clients: {clients}\n"
    else:
        hints.append(
            "HINT: Use a linker other than the C linker to"
            " print the inputs' shapes and strides."
        )

    # Print node backtraces
    tr = getattr(node.outputs[0].tag, "trace", [])
    if isinstance(tr, list) and len(tr) > 0:
        detailed_err_msg += "\nBacktrace when the node is created "
        detailed_err_msg += "(use Aesara flag traceback__limit=N to make it longer):\n"

        # Print separate message for each element in the list of batcktraces
        sio = io.StringIO()
        for subtr in tr:
            traceback.print_list(subtr, sio)
        detailed_err_msg += str(sio.getvalue())
    else:
        hints.append(
            "HINT: Re-running with most Aesara optimizations disabled could"
            " provide a back-trace showing when this node was created. This can"
            " be done by setting the Aesara flag"
            " 'optimizer=fast_compile'. If that does not work,"
            " Aesara optimizations can be disabled with 'optimizer=None'."
        )

    if verbosity == "high":

        import aesara.printing

        f = io.StringIO()
        aesara.printing.debugprint(node, file=f, stop_on_name=True, print_type=True)
        detailed_err_msg += "\nDebug print of the apply node: \n"
        detailed_err_msg += f.getvalue()

    # Prints output_map
    if verbosity == "high" and storage_map is not None:
        detailed_err_msg += "\nStorage map footprint:\n"
        shared_input_list = [
            item
            for item in fgraph.inputs
            if isinstance(item, aesara.compile.SharedVariable)
        ]
        nonshared_input_list = [
            item
            for item in fgraph.inputs
            if not isinstance(item, aesara.compile.SharedVariable)
        ]
        storage_map_list: List = []
        total_size = 0
        total_size_inputs = 0
        for k in storage_map:
            storage_map_item: List = []

            # storage_map_item[0]: the variable
            storage_map_item.append(str(k))

            # storage_map_item[1]: the shape
            shapeinfo = None
            if hasattr(storage_map[k][0], "shape"):
                shapeinfo = storage_map[k][0].shape
                if len(shapeinfo) != 0:
                    storage_map_item.append(shapeinfo)
                else:
                    storage_map_item.append(tuple())
            else:
                storage_map_item.append(None)

            # storage_map_item[2]: itemsize
            # storage_map_item[3]: bytes
            if hasattr(storage_map[k][0], "dtype"):
                dtype = storage_map[k][0].dtype
                storage_map_item.append(np.dtype(dtype).itemsize)
                if shapeinfo is None:
                    storage_map_item.append(-1)
                else:
                    sz = np.dtype(dtype).itemsize * np.prod(shapeinfo)
                    storage_map_item.append(sz)
                    total_size += sz
                    if not k.owner:
                        total_size_inputs += sz
                    else:
                        # If it is a view, don't count it twice.
                        if getattr(k.owner.op, "view_map", None):
                            vmap = k.owner.op.view_map
                            out_idx = k.owner.outputs.index(k)
                            data = storage_map[k][0]
                            if out_idx in vmap:
                                assert len(vmap[out_idx]) == 1
                                input_data = storage_map[
                                    k.owner.inputs[vmap[out_idx][0]]
                                ][0]
                                if k.type.may_share_memory(data, input_data):
                                    total_size -= sz
                        # If it is a destroyed input, the input
                        # shouldn't be in the storage_map anymore
                        # except if there is a special flag used. So
                        # we still must check it.
                        if getattr(k.owner.op, "destroy_map", None):
                            vmap = k.owner.op.destroy_map
                            out_idx = k.owner.outputs.index(k)
                            data = storage_map[k][0]
                            if out_idx in vmap:
                                assert len(vmap[out_idx]) == 1
                                input_data = storage_map[
                                    k.owner.inputs[vmap[out_idx][0]]
                                ][0]
                                if k.type.may_share_memory(data, input_data):
                                    total_size -= sz
            else:
                bytes = sys.getsizeof(storage_map[k][0])
                storage_map_item.append(bytes)
                storage_map_item.append(-1)

            # Flag of shared val
            # storage_map_item[4]
            if k in shared_input_list:
                storage_map_item.append(True)
            elif k in nonshared_input_list:
                storage_map_item.append(False)
            else:
                storage_map_item.append(None)
            storage_map_list.append(storage_map_item)

        storage_map_list.sort(key=itemgetter(3), reverse=True)
        for item in storage_map_list:
            if item[3] == -1:
                continue
            detailed_err_msg += " - " + item[0] + ", "
            if item[4] is True:
                detailed_err_msg += "Shared Input, "
            elif item[4] is False:
                detailed_err_msg += "Input, "
            if item[1] is not None:
                detailed_err_msg += f"Shape: {item[1]}, "
            detailed_err_msg += f"ElemSize: {item[2]} Byte(s)"
            if item[3] is not None:
                detailed_err_msg += f", TotalSize: {item[3]} Byte(s)\n"
            else:
                detailed_err_msg += "\n"
        detailed_err_msg += " TotalSize: {} Byte(s) {:.3f} GB\n".format(
            total_size,
            total_size / 1024 / 1024 / 1024,
        )
        detailed_err_msg += " TotalSize inputs: {} Byte(s) {:.3f} GB\n".format(
            total_size_inputs,
            total_size_inputs / 1024 / 1024 / 1024,
        )

    else:
        hints.append(
            "HINT: Use the Aesara flag `exception_verbosity=high`"
            " for a debug print-out and storage map footprint of this Apply node."
        )

    try:
        exc_value = exc_type(
            str(exc_value) + detailed_err_msg + "\n" + "\n".join(hints)
        )
    except TypeError:
        warnings.warn(
            f"{exc_type} error does not allow us to add an extra error message"
        )
        # Some exception need extra parameter in inputs. So forget the
        # extra long error message in that case.
    raise exc_value.with_traceback(exc_trace)


def __log_thunk_trace(value, handler: io.TextIOWrapper):
    """
    Log Aesara's diagnostic stack trace for an exception.

    Uses custom attributes that are added to trace objects by raise_with_op.
    """

    def write(msg):
        print(f"log_thunk_trace: {msg.strip()}", file=handler)

    if hasattr(value, "__thunk_trace__"):
        trace2 = value.__thunk_trace__
        write("There was a problem executing an Op.")
        if trace2 is None:
            write("Could not find where this Op was defined.")
            write(
                " * You might have instantiated this Op "
                "directly instead of using a constructor."
            )
            write(
                " * The Op you constructed might have been"
                " optimized. Try turning off optimizations."
            )
        elif trace2:
            write("Definition in: ")
            for line in traceback.format_list(trace2):
                write(line)
            write(
                "For the full definition stack trace set"
                " the Aesara flags `traceback__limit` to -1"
            )


def register_thunk_trace_excepthook(handler: io.TextIOWrapper = sys.stdout):
    """Adds the __log_thunk_trace except hook to the collection in aesara.utils.

    Parameters
    ----------
    handler : TextIOWrapper
        Target for printing the output.
    """

    def wrapper(type, value, trace):
        __log_thunk_trace(value, handler)

    utils.add_excepthook(wrapper)


register_thunk_trace_excepthook()

import io
import sys
import traceback
import warnings
from operator import itemgetter

import numpy as np

from theano import config, utils
from theano.gof.fg import FunctionGraph


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
        storage map of the theano function that resulted in the
        raised exception.

    Notes
    -----
    This re-raises the exception described by `exc_info` (or the last
    one raised, if `exc_info` is omitted) and annotates the exception
    object with several new members which may be helpful for debugging
    Theano graphs. They are:

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
            shapes = "The thunk don't have an inputs attributes."
            strides = "So we can't access the strides of inputs values"
            scalar_values = "And can't print its inputs scalar value"
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
            "HINT: Use another linker then the c linker to"
            " have the inputs shapes and strides printed."
        )

    # Print node backtraces
    tr = getattr(node.outputs[0].tag, "trace", [])
    if isinstance(tr, list) and len(tr) > 0:
        detailed_err_msg += "\nBacktrace when the node is created(use Theano flag traceback__limit=N to make it longer):\n"

        # Print separate message for each element in the list of batcktraces
        sio = io.StringIO()
        for subtr in tr:
            traceback.print_list(subtr, sio)
        detailed_err_msg += str(sio.getvalue())
    else:
        hints.append(
            "HINT: Re-running with most Theano optimization disabled could"
            " give you a back-trace of when this node was created. This can"
            " be done with by setting the Theano flag"
            " 'optimizer=fast_compile'. If that does not work,"
            " Theano optimizations can be disabled with 'optimizer=None'."
        )

    if verbosity == "high":

        import theano.printing

        f = io.StringIO()
        theano.printing.debugprint(node, file=f, stop_on_name=True, print_type=True)
        detailed_err_msg += "\nDebugprint of the apply node: \n"
        detailed_err_msg += f.getvalue()

    # Prints output_map
    if verbosity == "high" and storage_map is not None:
        detailed_err_msg += "\nStorage map footprint:\n"
        shared_input_list = [
            item
            for item in fgraph.inputs
            if isinstance(item, theano.compile.SharedVariable)
        ]
        nonshared_input_list = [
            item
            for item in fgraph.inputs
            if not isinstance(item, theano.compile.SharedVariable)
        ]
        storage_map_list = []
        total_size = 0
        total_size_inputs = 0
        for k in storage_map:
            storage_map_item = []

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
            "HINT: Use the Theano flag 'exception_verbosity=high'"
            " for a debugprint and storage map footprint of this apply node."
        )

    try:
        exc_value = exc_type(
            str(exc_value) + detailed_err_msg + "\n" + "\n".join(hints)
        )
    except TypeError:
        warnings.warn(f"{exc_type} error does not allow us to add extra error message")
        # Some exception need extra parameter in inputs. So forget the
        # extra long error message in that case.
    raise exc_value.with_traceback(exc_trace)


def __log_thunk_trace(value, handler: io.TextIOWrapper):
    """
    Log Theano's diagnostic stack trace for an exception.

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
                " the Theano flags traceback__limit to -1"
            )


def register_thunk_trace_excepthook(handler: io.TextIOWrapper = sys.stdout):
    """Adds the __log_thunk_trace except hook to the collection in theano.utils.

    Parameters
    ----------
    handler : TextIOWrapper
        Target for printing the output.
    """

    def wrapper(type, value, trace):
        __log_thunk_trace(value, handler)

    utils.add_excepthook(wrapper)


register_thunk_trace_excepthook()

"""
Provide a simple user friendly API.

"""

import logging
from copy import copy
from typing import Optional

from aesara.compile.function.types import Function, UnusedInputError, orig_function
from aesara.compile.io import In, Out
from aesara.compile.profiling import ProfileStats
from aesara.compile.sharedvalue import SharedVariable, shared
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable, clone_node_and_cache
from aesara.graph.fg import FunctionGraph


_logger = logging.getLogger("aesara.compile.function.pfunc")

__docformat__ = "restructuredtext en"


def rebuild_collect_shared(
    outputs,
    inputs=None,
    replace=None,
    updates=None,
    rebuild_strict=True,
    copy_inputs_over=True,
    no_default_updates=False,
    clone_inner_graphs=False,
):
    r"""Replace subgraphs of a computational graph.

    It returns a set of dictionaries and lists which collect (partial?)
    different information about shared variables. This info is required by
    `pfunc`.

    Parameters
    ----------
    outputs : list of Aesara Variables (or Aesara expressions)
        List of Aesara variables or expressions representing the outputs of the
        computational graph.
    inputs : list of Aesara Variables (or Aesara expressions)
        List of Aesara variables or expressions representing the inputs of the
        computational graph (or None).
    replace : dict
        Dictionary describing which subgraphs should be replaced by what.
        orig_value => new_value
    updates : dict
        Dictionary describing updates expressions for shared variables.
    rebuild_strict : bool
        Flag, if true the type of all inputs should be the same as the one for
        the current node.
    copy_inputs_over : bool
        Flag; if False it will clone inputs.
    no_default_updates : either bool or list of Variables
        If True, do not perform any automatic update on Variables.
        If False (default), perform them all.
        Else, perform automatic updates on all Variables that are neither in
        "updates" nor in "no_default_updates".
    clone_inner_graphs : bool
        If ``True``, clone `Op`\s that are subclasses of `HasInnerGraph` and their
        inner-graphs.

    """

    if isinstance(outputs, tuple):
        outputs = list(outputs)

    # This function implements similar functionality as graph.clone
    # and it should be merged with that
    clone_d = {}
    update_d = {}
    update_expr = []
    # list of shared inputs that are used as inputs of the graph
    shared_inputs = []

    def clone_v_get_shared_updates(v, copy_inputs_over):
        """
        Clones a variable and its inputs recursively until all are in clone_d.
        Also appends all shared variables met along the way to shared inputs,
        and their default_update (if applicable) to update_d and update_expr.

        """
        # this co-recurses with clone_a
        assert v is not None
        if v in clone_d:
            return clone_d[v]
        if v.owner:
            owner = v.owner
            if owner not in clone_d:
                for i in owner.inputs:
                    clone_v_get_shared_updates(i, copy_inputs_over)
                clone_node_and_cache(
                    owner,
                    clone_d,
                    strict=rebuild_strict,
                    clone_inner_graphs=clone_inner_graphs,
                )
            return clone_d.setdefault(v, v)
        elif isinstance(v, SharedVariable):
            if v not in shared_inputs:
                shared_inputs.append(v)
            if hasattr(v, "default_update"):
                # Check that v should not be excluded from the default
                # updates list
                if no_default_updates is False or (
                    isinstance(no_default_updates, list) and v not in no_default_updates
                ):
                    # Do not use default_update if a "real" update was
                    # provided
                    if v not in update_d:
                        v_update = v.type.filter_variable(
                            v.default_update, allow_convert=False
                        )
                        if not v.type.is_super(v_update.type):
                            raise TypeError(
                                "An update must have a type compatible with "
                                "the original shared variable"
                            )
                        update_d[v] = v_update
                        update_expr.append((v, v_update))
        if not copy_inputs_over:
            return clone_d.setdefault(v, v.clone())
        else:
            return clone_d.setdefault(v, v)

    # initialize the clone_d mapping with the replace dictionary
    if replace is None:
        replace = []
    try:
        replace_pairs = list(replace.items())
    except Exception:
        replace_pairs = replace

    for v_orig, v_repl in replace_pairs:
        if not isinstance(v_orig, Variable):
            raise TypeError("`givens` keys must be Variables")
        if not isinstance(v_repl, Variable):
            v_repl = shared(v_repl)

        if v_orig in clone_d:
            raise AssertionError(
                "When using 'givens' or 'replace' with several "
                "(old_v, new_v) replacement pairs, you can not have a "
                "new_v variable depend on an old_v one. For instance, "
                "givens = {a:b, b:(a+1)} is not allowed. Here, the old_v "
                f"{v_orig} is used to compute other new_v's, but it is scheduled "
                f"to be replaced by {v_repl}."
            )

        clone_d[v_orig] = clone_v_get_shared_updates(v_repl, copy_inputs_over)

    if inputs is None:
        inputs = []

    def clone_inputs(i):
        if not copy_inputs_over:
            return clone_d.setdefault(i, i.clone())
        else:
            return clone_d.setdefault(i, i)

    input_variables = [clone_inputs(i) for i in inputs]

    # It was decided, as a first step, to prevent shared variables from
    # being used as function inputs. Although it is technically possible,
    # it is also not clear when/how to use the value of that shared
    # variable (is it a default? ignored?, if the shared variable changes,
    # does that function default also change?).
    for v in input_variables:
        if isinstance(v, SharedVariable):
            raise TypeError(
                f"Cannot use a shared variable ({v}) as explicit "
                "input. Consider substituting a non-shared"
                " variable via the `givens` parameter"
            )

    # Fill update_d and update_expr with provided updates
    if updates is None:
        updates = []
    for (store_into, update_val) in iter_over_pairs(updates):
        if not isinstance(store_into, SharedVariable):
            raise TypeError("update target must be a SharedVariable", store_into)
        if store_into in update_d:
            raise ValueError(
                "this shared variable already has an update " "expression",
                (store_into, update_d[store_into]),
            )

        # filter_variable ensure smooth conversion of cpu Types
        try:
            update_val = store_into.type.filter_variable(
                update_val, allow_convert=False
            )
        except TypeError:
            err_msg = (
                "An update must have the same type as the"
                f" original shared variable (shared_var={store_into},"
                f" shared_var.type={store_into.type},"
                f" update_val={update_val}, update_val.type={getattr(update_val, 'type', None)})."
            )
            err_sug = (
                "If the difference is related to the broadcast pattern,"
                " you can call the"
                " tensor.shape.unbroadcast(var, axis_to_unbroadcast[, ...])"
                " function to mask broadcastable dimensions."
            )

            raise TypeError(err_msg, err_sug)
        assert store_into.type.is_super(update_val.type)

        update_d[store_into] = update_val
        update_expr.append((store_into, update_val))

    # Elements of "outputs" are here cloned to "cloned_outputs"
    if isinstance(outputs, list):
        cloned_outputs = []
        for v in outputs:
            if isinstance(v, Variable):
                cloned_v = clone_v_get_shared_updates(v, copy_inputs_over)
                cloned_outputs.append(cloned_v)
            elif isinstance(v, Out):
                cloned_v = clone_v_get_shared_updates(v.variable, copy_inputs_over)
                cloned_outputs.append(Out(cloned_v, borrow=v.borrow))
            else:
                raise TypeError(
                    "Outputs must be aesara Variable or "
                    "Out instances. Received " + str(v) + " of type " + str(type(v))
                )
            # computed_list.append(cloned_v)
    else:
        if isinstance(outputs, Variable):
            cloned_v = clone_v_get_shared_updates(outputs, copy_inputs_over)
            cloned_outputs = cloned_v
            # computed_list.append(cloned_v)
        elif isinstance(outputs, Out):
            cloned_v = clone_v_get_shared_updates(outputs.variable, copy_inputs_over)
            cloned_outputs = Out(cloned_v, borrow=outputs.borrow)
            # computed_list.append(cloned_v)
        elif outputs is None:
            cloned_outputs = []  # TODO: get Function.__call__ to return None
        else:
            raise TypeError(
                "output must be an Aesara Variable or Out "
                "instance (or list of them)",
                outputs,
            )

    # Iterate over update_expr, cloning its elements, and updating
    # shared_inputs, update_d and update_expr from the SharedVariables
    # we discover.
    # If the variable to be updated is a shared variable not already
    # in shared_inputs, add it.
    # Note: we extend update_expr while iterating over it.

    i = 0
    while i < len(update_expr):
        v, v_update = update_expr[i]
        cloned_v_update = clone_v_get_shared_updates(v_update, copy_inputs_over)
        update_d[v] = cloned_v_update
        if isinstance(v, SharedVariable) and v not in shared_inputs:
            shared_inputs.append(v)
        i += 1

    return (
        input_variables,
        cloned_outputs,
        [clone_d, update_d, update_expr, shared_inputs],
    )


def pfunc(
    params,
    outputs=None,
    mode=None,
    updates=None,
    givens=None,
    no_default_updates=False,
    accept_inplace=False,
    name=None,
    rebuild_strict=True,
    allow_input_downcast=None,
    profile=None,
    on_unused_input=None,
    output_keys=None,
    fgraph: Optional[FunctionGraph] = None,
) -> Function:
    """
    Function-constructor for graphs with shared variables.

    Parameters
    ----------
    params : list of either Variable or In instances
        Function parameters, these are not allowed to be shared variables.
    outputs : list of Variables or Out instances
        Expressions to compute.
    mode : string or `aesara.compile.mode.Mode` instance
        Compilation mode.
    updates : iterable over pairs (shared_variable, new_expression). List, tuple or dict.
        Update the values for SharedVariable inputs according to these
        expressions
    givens : iterable over pairs (Var1, Var2) of Variables. List, tuple or dict.
        The Var1 and Var2 in each pair must have the same Type. Specific
        substitutions to make in the computation graph (Var2 replaces Var1).
    no_default_updates : either bool or list of Variables
        If True, do not perform any automatic update on Variables.
        If False (default), perform them all. Else, perform automatic updates
        on all Variables that are neither in "updates" nor in
        "no_default_updates".
    accept_inplace : bool
        True iff the graph can contain inplace operations prior to the
        optimization phase (default is False). *Note* this parameter is unsupported,
        and its use is not recommended.
    name : None or string
        Attaches a name to the profiling result of this function.
    allow_input_downcast : bool
        True means that the values passed as inputs when calling the function
        can be silently downcasted to fit the dtype of the corresponding
        Variable, which may lose precision. False means that it will only be cast to a more
        general, or precise, type. None (default) is almost like
        False, but allows downcasting of Python float scalars to
        floatX.
    profile : None, True, str, or ProfileStats instance
        Accumulate profiling information into a given ProfileStats instance.
        None is the default, and means to use the value of config.profile.
        If argument is `True` then a new ProfileStats instance will be used.
        If argument is a string, a new ProfileStats instance will be created
        with that string as its `message` attribute. This profiling object will
        be available via self.profile.
    on_unused_input : {'raise', 'warn','ignore', None}
        What to do if a variable in the 'inputs' list is not used in the graph.
    fgraph
        An existing `FunctionGraph` from which to construct the returned
        `Function`.  When this is non-``None``, nothing is cloned.

    Returns
    -------
    A callable object that will compute the outputs (given the inputs) and
    update the implicit function arguments according to the `updates`.

    Notes
    -----
    Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when ``Var1`` of one pair appears in the graph leading
    to ``Var2`` in another expression is undefined. Replacements specified with
    givens are different from optimizations in that ``Var2`` is not expected to
    be equivalent to ``Var1``.

    """

    if profile is None:
        profile = config.profile or config.print_global_stats
        if profile is False:
            profile = None
    if profile is True:
        profile = ProfileStats(message=name)
    elif isinstance(profile, str):
        profile = ProfileStats(message=profile)

    inputs, cloned_outputs = construct_pfunc_ins_and_outs(
        params,
        outputs,
        mode,
        updates,
        givens,
        no_default_updates,
        rebuild_strict,
        allow_input_downcast,
        fgraph=fgraph,
    )

    return orig_function(
        inputs,
        cloned_outputs,
        mode,
        accept_inplace=accept_inplace,
        name=name,
        profile=profile,
        on_unused_input=on_unused_input,
        output_keys=output_keys,
        fgraph=fgraph,
    )


def construct_pfunc_ins_and_outs(
    params,
    outputs=None,
    mode=None,
    updates=None,
    givens=None,
    no_default_updates=False,
    rebuild_strict=True,
    allow_input_downcast=None,
    fgraph: Optional[FunctionGraph] = None,
):
    """Construct inputs and outputs for `pfunc`.

    This function works by cloning the graph (except for the
    inputs), and then shipping it off to aesara.compile.function.function
    (There it will be cloned again, unnecessarily, because it doesn't know
    that we already cloned it.)

    First, it clones the replacements named in the `givens` argument,
    and points each ``Var1`` to the clone of ``Var2``.  Then it sets the
    inputs in the clone dictionary.  After these steps, we are
    assuming that the clone dictionary contains all the inputs to
    the computation graph.

    Then it clones the outputs and the update expressions.  This
    rebuilds a computation graph from the inputs and the `givens`.

    When `fgraph` is non-``None``, nothing is cloned and the given `fgraph` is
    simply prepared for direct use.

    """
    if updates is None:
        updates = []

    if givens is None:
        givens = []

    if not isinstance(params, (list, tuple)):
        raise Exception("in pfunc() the first argument must be a list or " "a tuple")

    if not isinstance(no_default_updates, bool) and not isinstance(
        no_default_updates, list
    ):
        raise TypeError("no_default_update should be either a boolean or " "a list")

    if len(updates) > 0 and any(
        isinstance(v, Variable) for v in iter_over_pairs(updates)
    ):
        raise ValueError(
            "The updates parameter must be an OrderedDict/dict or a list of "
            "lists/tuples with 2 elements"
        )

    # transform params into aesara.compile.In objects.
    inputs = [
        _pfunc_param_to_in(p, allow_downcast=allow_input_downcast) for p in params
    ]

    # Check if some variable is present more than once in inputs
    in_variables = [input.variable for input in inputs]
    for i, v in enumerate(in_variables):
        if v in in_variables[(i + 1) :]:
            dup_v_i = in_variables.index(v, (i + 1))
            raise UnusedInputError(
                f"Variable {v} is used twice in inputs to aesara.function, "
                f"at indices {i} and {dup_v_i}.  This would result in values "
                "provided for it being ignored. Please do not duplicate "
                "variables in the inputs list."
            )

    # Check that we are not using `givens` to replace input variables, because
    # this typically does nothing, contrary to what one may expect.
    in_var_set = set(in_variables)
    try:
        givens_pairs = list(givens.items())
    except AttributeError:
        givens_pairs = givens
    for x, y in givens_pairs:
        if x in in_var_set:
            raise RuntimeError(
                f"You are trying to replace variable '{x}' through the "
                "`givens` parameter, but this variable is an input to your "
                "function. Replacing inputs is currently forbidden because it "
                "has no effect. One way to modify an input `x` to a function "
                "evaluating f(x) is to define a new input `y` and use "
                "`aesara.function([y], f(x), givens={x: g(y)})`. Another "
                "solution consists in using `aesara.clone_replace`, e.g. like this: "
                "`aesara.function([x], "
                "aesara.clone_replace(f(x), replace={x: g(x)}))`."
            )

    if not fgraph:

        # Extend the outputs with the updates on input variables so they are
        # also cloned
        additional_outputs = [i.update for i in inputs if i.update]
        if outputs is None:
            out_list = []
        else:
            if isinstance(outputs, (list, tuple)):
                out_list = list(outputs)
            else:
                out_list = [outputs]
        extended_outputs = out_list + additional_outputs

        output_vars = rebuild_collect_shared(
            extended_outputs,
            in_variables,
            replace=givens,
            updates=updates,
            rebuild_strict=rebuild_strict,
            copy_inputs_over=True,
            no_default_updates=no_default_updates,
            clone_inner_graphs=True,
        )
        input_variables, cloned_extended_outputs, other_stuff = output_vars
        clone_d, update_d, update_expr, shared_inputs = other_stuff

        # Recover only the clones of the original outputs
        if outputs is None:
            new_outputs = []
        else:
            if isinstance(outputs, (list, tuple)):
                new_outputs = cloned_extended_outputs[: len(outputs)]
            else:
                new_outputs = cloned_extended_outputs[0]

        new_inputs = []

        for i, iv in zip(inputs, input_variables):
            new_i = copy(i)
            new_i.variable = iv

            # If needed, replace the input's update by its cloned equivalent
            if i.update:
                new_i.update = clone_d[i.update]

            new_inputs.append(new_i)

        for sv in shared_inputs:
            if sv in update_d:
                si = In(
                    variable=sv,
                    value=sv.container,
                    mutable=True,
                    borrow=True,
                    update=update_d[sv],
                    shared=True,
                )
            else:
                si = In(
                    variable=sv,
                    value=sv.container,
                    mutable=False,
                    borrow=True,
                    shared=True,
                )
            new_inputs.append(si)

    else:
        assert len(fgraph.inputs) == len(inputs)
        assert len(fgraph.outputs) == len(outputs)

        for fg_inp, inp in zip(fgraph.inputs, inputs):
            if fg_inp != getattr(inp, "variable", inp):
                raise ValueError(
                    f"`fgraph`'s input does not match the provided input: {fg_inp}, {inp}"
                )

        for fg_out, out in zip(fgraph.outputs, outputs):
            if fg_out != getattr(out, "variable", out):
                raise ValueError(
                    f"`fgraph`'s output does not match the provided output: {fg_out}, {out}"
                )

        new_inputs = inputs
        new_outputs = outputs

    return new_inputs, new_outputs


def _pfunc_param_to_in(param, strict=False, allow_downcast=None):
    if isinstance(param, Constant):
        raise TypeError("Constants not allowed in param list", param)
    if isinstance(param, Variable):  # N.B. includes SharedVariable
        return In(variable=param, strict=strict, allow_downcast=allow_downcast)
    elif isinstance(param, In):
        return param
    raise TypeError(f"Unknown parameter type: {type(param)}")


def iter_over_pairs(pairs):
    """
    Return an iterator over pairs present in the 'pairs' input.

    Parameters
    ----------
    pairs : dictionary or iterable
        The pairs to iterate upon. These may be stored either as (key, value)
        items in a dictionary, or directly as pairs in any kind of iterable
        structure.

    Returns
    -------
    iterable
        An iterable yielding pairs.

    """
    if isinstance(pairs, dict):
        return pairs.items()
    else:
        return pairs

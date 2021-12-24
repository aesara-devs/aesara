"""A container for specifying and manipulating a graph with distinct inputs and outputs."""
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import aesara
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant, Variable, applys_between
from aesara.graph.basic import as_string as graph_as_string
from aesara.graph.basic import clone_get_equiv, graph_inputs, io_toposort, vars_between
from aesara.graph.features import AlreadyThere, Feature, ReplaceValidate
from aesara.graph.utils import MetaObject, TestValueError, get_variable_trace_string
from aesara.misc.ordered_set import OrderedSet


class InconsistencyError(Exception):
    """
    This exception should be thrown by listeners to FunctionGraph when the
    graph's state is invalid.

    """


class MissingInputError(Exception):
    """
    A symbolic input needed to compute the outputs is missing.

    """

    def __init__(self, *args, **kwargs):
        if kwargs:
            # The call to list is needed for Python 3
            assert list(kwargs.keys()) == ["variable"]
            error_msg = get_variable_trace_string(kwargs["variable"])
            if error_msg:
                args = args + (error_msg,)
        s = "\n".join(args)  # Needed to have the new line print correctly
        super().__init__(s)


class FunctionGraph(MetaObject):
    """
    A `FunctionGraph` represents a subgraph bound by a set of input variables and
    a set of output variables, ie a subgraph that specifies an Aesara function.
    The inputs list should contain all the inputs on which the outputs depend.
    ``Variable``s of type ``Constant`` are not counted as inputs.

    The `FunctionGraph` supports the replace operation which allows to replace
    a variable in the subgraph by another, e.g. replace ``(x + x).out`` by
    ``(2 * x).out``. This is the basis for optimization in Aesara.

    This class is also responsible for verifying that a graph is valid
    (ie, all the dtypes and broadcast patterns are compatible with the
    way the ``Variable``s are used) and for tracking the ``Variable``s with
    a ``clients`` field that specifies which ``Apply`` nodes use the ``Variable``.
    The ``clients`` field combined with the ``Variable.owner`` field and the
    ``Apply`` nodes' ``Apply.inputs`` field allows the graph to be traversed in
    both directions.

    It can also be extended with new features using
    ``FunctionGraph.attach_feature(<Feature instance>)``.
    See ``Feature`` for event types and documentation.
    Extra features allow the `FunctionGraph` to verify new properties of
    a graph as it is optimized.

    Historically, the `FunctionGraph` was called an ``Env``. Keep this in mind
    while reading out-of-date documentation, e-mail support threads, etc.

    The constructor creates a `FunctionGraph` which operates on the subgraph
    bound by the inputs and outputs sets.

    This class keeps a pointer to the inputs and outputs, and also modifies
    them.

    """

    def __init__(
        self,
        inputs: Optional[List[Variable]] = None,
        outputs: Optional[List[Variable]] = None,
        features: Optional[List[Feature]] = None,
        clone: bool = False,
        update_mapping: Optional[Dict[Variable, Variable]] = None,
        memo: Optional[Dict[Variable, Variable]] = None,
        copy_inputs: bool = True,
        copy_orphans: bool = True,
    ):
        """
        Create a `FunctionGraph` which operates on the subgraph between the
        `inputs` and `outputs`.

        Parameters
        ----------
        inputs
            Input variables of the graph.
        outputs
            Output variables of the graph.
        clone
            If ``True``, the graph will be cloned.
        features
            A list of features to be added to the `FunctionGraph`.
        update_mapping
            Mapping between the `inputs` with updates and the `outputs`
            corresponding to their updates.
        memo
            See ``clone_get_equiv``.
        copy_inputs
            See ``clone_get_equiv``.
        copy_orphans
            See ``clone_get_equiv``.
        """
        if outputs is None:
            raise ValueError("No outputs specified")

        if inputs is None:
            inputs = [i for i in graph_inputs(outputs) if not isinstance(i, Constant)]

        if clone:
            memo = clone_get_equiv(
                inputs,
                outputs,
                copy_inputs=copy_inputs,
                copy_orphans=copy_orphans,
                memo=memo,
            )
            outputs = [memo[o] for o in outputs]
            inputs = [memo[i] for i in inputs]

        self.execute_callbacks_time = 0
        self.execute_callbacks_times = {}

        if features is None:
            features = []

        self._features = []

        # All apply nodes in the subgraph defined by inputs and
        # outputs are cached in this field
        self.apply_nodes = set()

        # Ditto for variable nodes.
        # It must contain all fgraph.inputs and all apply_nodes
        # outputs even if they aren't used in the graph.
        self.variables = set()

        self.inputs = []
        self.outputs = list(outputs)
        self.clients = {}

        for f in features:
            self.attach_feature(f)

        self.attach_feature(ReplaceValidate())

        for in_var in inputs:
            # if in_var.owner is not None:
            #     raise ValueError(
            #         "One of the provided inputs is the output of "
            #         "an already existing node. "
            #         "If that is okay, either discard that "
            #         "input's owner or use graph.clone."
            #     )
            self.add_input(in_var, check=False)

        for output in outputs:
            self.import_var(output, reason="init")
        # for i, output in enumerate(outputs):
        #     self.clients[output].append(("output", i))

        self.profile = None
        self.update_mapping = update_mapping

    def add_input(self, var: Variable, check: bool = True) -> None:
        """Add a new variable as an input to this `FunctionGraph`.

        Parameters
        ----------
        var : aesara.graph.basic.Variable

        """
        if check and var in self.inputs:
            return

        self.inputs.append(var)
        self.setup_var(var)
        self.variables.add(var)

    def setup_var(self, var: Variable) -> None:
        """Set up a variable so it belongs to this `FunctionGraph`.

        Parameters
        ----------
        var : aesara.graph.basic.Variable

        """
        self.clients.setdefault(var, [])

    def setup_node(self, node: Apply) -> None:
        """Set up node so it belongs to this `FunctionGraph`.

        Parameters
        ----------
        node : aesara.graph.basic.Apply

        """
        if node.op.view_map and not all(
            isinstance(view, (list, tuple)) for view in node.op.view_map.values()
        ):
            raise Exception(
                f"Op '{node.op}' has a bad view map '{node.op.view_map}',"
                " the values must be tuples or lists."
            )
        if node.op.destroy_map and not all(
            isinstance(destroy, (list, tuple))
            for destroy in node.op.destroy_map.values()
        ):
            raise Exception(
                f"Op '{node.op}' has a bad destroy map '{node.op.destroy_map}',"
                " the values must be tuples or lists."
            )

    def disown(self) -> None:
        """Clear internal variables."""
        for f in self._features:
            self.remove_feature(f)
        self.clients = {}
        self.apply_nodes = set()
        self.variables = set()
        self.inputs = None
        self.outputs = None
        self.profile = None
        self.update_mapping = None

    def get_clients(self, var: Variable) -> List[Tuple[Apply, int]]:
        """Return a list of all the `(node, i)` pairs such that `node.inputs[i]` is `var`."""
        return self.clients[var]

    def add_client(self, var: Variable, new_client: Tuple[Apply, int]) -> None:
        """Update the clients of `var` with `new_clients`.

        Parameters
        ----------
        var : Variable
            The `Variable` to be updated.
        new_client : (Apply, int)
            A ``(node, i)`` pair such that ``node.inputs[i]`` is `var`.

        """
        self.clients[var].append(new_client)

    def create_next_nodes_memo(self, node, new_node, i, memo, old_clients):
        # This will add the new node as mapping to memo object

        memo[node.out] = new_node.out

        if not old_clients[node.out]:
            # This is an output variable
            assert node.out in self.outputs

        curr_clients = old_clients[node.out]

        for old_node, idx in curr_clients:
            # For every old clients of current node, make a new node
            # by along with changed inputs and calling the function
            # recursively.
            curr_node = (
                memo[old_node.out].owner if old_node.out in memo.keys() else old_node
            )
            next_new_inputs = list(curr_node.inputs)
            for _idx, _var in enumerate(next_new_inputs):
                if _var and _var in memo.keys():
                    next_new_inputs[_idx] = memo[_var]
            next_new_inputs[idx] = new_node.out
            next_new_node = curr_node.op.make_node(*next_new_inputs)
            memo.update(
                self.create_next_nodes_memo(
                    old_node, next_new_node, idx, memo, old_clients
                )
            )

        return memo

    def _get_unused_var_inputs(self, node):
        unused_vars = set()
        curr_inputs = node.inputs
        for i, _input in enumerate(curr_inputs):
            self.clients[_input].remove((node, i))
            if not self.clients[_input] and _input not in self.outputs:
                unused_vars.add(_input)
                if _input.owner:
                    unused_vars.update(self._get_unused_var_inputs(_input.owner))
        return unused_vars

    def _replace_nodes(self, pairs_to_replace, old_clients, memo, reason):
        def replace_clients(_old_var, _new_var):
            self.clients.pop(_old_var)
            # The new variable might already be part of FunctionGraph.
            if _new_var not in self.clients.keys():
                self.setup_var(_new_var)
            for node, idx in old_clients[_old_var]:
                # If its a client it must have an owner and inputs
                curr_node = memo[node.out].owner
                for i, _inp in enumerate(node.inputs):
                    # If an input is not in memo that means it's
                    # an unchanged variable, in that case we
                    # need to update the node in it's clients.
                    if (
                        _inp not in memo.keys()
                        and (curr_node, i) not in self.clients[_inp]
                    ):
                        self.clients[_inp].remove((node, i))
                        self.clients[_inp].append((curr_node, i))
                self.clients[_new_var].append((curr_node, idx))
                # If node.out is not in self.clients.keys() that means
                # it is either an output or has already been processed
                # due to it being output of some other variable.
                if node.out in self.clients.keys():
                    replace_clients(node.out, curr_node.out)

        for old_var, new_var in pairs_to_replace.items():
            replace_clients(old_var, new_var)

        # We update the veiables, apply nodes, inputs and outputs
        # according to memo
        for old_var, new_var in memo.items():
            self.variables.remove(old_var)
            self.variables.add(new_var)
            if old_var.owner:
                self.apply_nodes.remove(old_var.owner)
            if new_var.owner:
                self.apply_nodes.add(new_var.owner)
            if old_var in self.inputs:
                self.inputs[self.inputs.index(old_var)] = new_var
            elif old_var in self.outputs:
                self.outputs[self.outputs.index(old_var)] = new_var

        # The following logic is to determine and remove any variables that are no
        # longer used in the FunctionGraph i.e they were only used to calculate
        # the old variables but not the new ones.
        unused_vars = set()
        for var, new_var in pairs_to_replace.items():
            unused_vars.add(var)
            if var.owner:
                unused_vars.update(self._get_unused_var_inputs(var.owner))

        for var in list(unused_vars):
            if var in self.clients.keys():
                if self.clients[var] or var in self.outputs:
                    unused_vars.remove(var)
                    continue
                else:
                    self.clients.pop(var)
            if var in self.variables:
                self.variables.remove(var)
            if var.owner and var.owner in self.apply_nodes:
                self.execute_callbacks("on_prune", var.owner, reason)
                self.apply_nodes.remove(var.owner)

        return unused_vars

    def remove_client(
        self, var: Variable, client_to_remove: Tuple[Apply, int], reason: str = None
    ) -> None:
        """Recursively remove clients of a variable.

        This is the main method to remove variables or `Apply` nodes from
        a `FunctionGraph`.

        This will remove `var` from the `FunctionGraph` if it doesn't have any
        clients remaining. If it has an owner and all the outputs of the owner
        have no clients, it will also be removed.

        Parameters
        ----------
        var : Variable
            The clients of `var` that will be removed.
        client_to_remove : pair of (Apply, int)
            A ``(node, i)`` pair such that ``node.inputs[i]`` will no longer be
            `var` in this `FunctionGraph`.

        """

        removal_stack = [(var, client_to_remove)]
        while removal_stack:
            var, client_to_remove = removal_stack.pop()

            try:
                var_clients = self.clients[var]
                var_clients.remove(client_to_remove)
            except ValueError:
                # In this case, the original `var` could've been removed from
                # the current `var`'s client list before this call.
                # There's nothing inherently wrong with that, so we continue as
                # if it were removed here.
                var_clients = None

            if var_clients or (var in self.outputs):
                continue

            # Now, `var` has no more clients, so check if we need to remove it
            # and its `Apply` node
            if not var.owner:
                # The `var` is a `Constant` or an input without a client, so we
                # remove it
                self.variables.remove(var)
            else:
                apply_node = var.owner
                if not any(
                    output for output in apply_node.outputs if self.clients[output]
                ):
                    # The `Apply` node is not used and is not an output, so we
                    # remove it and its outputs
                    if not hasattr(apply_node.tag, "removed_by"):
                        apply_node.tag.removed_by = []

                    apply_node.tag.removed_by.append(str(reason))

                    self.apply_nodes.remove(apply_node)

                    self.variables.difference_update(apply_node.outputs)

                    self.execute_callbacks("on_prune", apply_node, reason)

                    for i, in_var in enumerate(apply_node.inputs):
                        removal_stack.append((in_var, (apply_node, i)))

    def import_var(
        self, var: Variable, reason: str = None, import_missing: bool = False
    ) -> None:
        """Import variables into this `FunctionGraph`.

        This will also import the `variable`'s `Apply` node.

        Parameters
        ----------
        variable : aesara.graph.basic.Variable
            The variable to be imported.
        reason : str
            The name of the optimization or operation in progress.
        import_missing : bool
            Add missing inputs instead of raising an exception.

        """
        # Imports the owners of the variables
        if var.owner and var.owner not in self.apply_nodes:
            self.import_node(var.owner, reason=reason, import_missing=import_missing)
        elif (
            var.owner is None
            and not isinstance(var, Constant)
            and var not in self.inputs
        ):
            from aesara.graph.null_type import NullType

            if isinstance(var.type, NullType):
                raise TypeError(
                    f"Computation graph contains a NaN. {var.type.why_null}"
                )
            if import_missing:
                self.add_input(var)
            else:
                raise MissingInputError(f"Undeclared input: {var}", variable=var)
        self.setup_var(var)
        self.variables.add(var)

    def import_node(
        self,
        apply_node: Apply,
        check: bool = True,
        reason: str = None,
        import_missing: bool = False,
    ) -> None:
        """Recursively import everything between an ``Apply`` node and the ``FunctionGraph``'s outputs.

        Parameters
        ----------
        apply_node : Apply
            The node to be imported.
        check : bool
            Check that the inputs for the imported nodes are also present in
            the `FunctionGraph`.
        reason : str
            The name of the optimization or operation in progress.
        import_missing : bool
            Add missing inputs instead of raising an exception.
        """
        # We import the nodes in topological order. We only are interested in
        # new nodes, so we use all variables we know of as if they were the
        # input set.  (The functions in the graph module only use the input set
        # to know where to stop going down.)
        new_nodes = io_toposort(self.variables, apply_node.outputs)

        if check:
            for node in new_nodes:
                for var in node.inputs:
                    if (
                        var.owner is None
                        and not isinstance(var, Constant)
                        and var not in self.inputs
                    ):
                        if import_missing:
                            self.add_input(var)
                        else:
                            error_msg = (
                                f"Input {node.inputs.index(var)} ({var})"
                                " of the graph (indices start "
                                f"from 0), used to compute {node}, was not "
                                "provided and not given a value. Use the "
                                "Aesara flag exception_verbosity='high', "
                                "for more information on this error."
                            )
                            raise MissingInputError(error_msg, variable=var)

        for node in new_nodes:
            assert node not in self.apply_nodes
            self.setup_node(node)
            self.apply_nodes.add(node)
            if not hasattr(node.tag, "imported_by"):
                node.tag.imported_by = []
            node.tag.imported_by.append(str(reason))
            for output in node.outputs:
                self.setup_var(output)
                self.variables.add(output)
            for i, input in enumerate(node.inputs):
                if input not in self.variables:
                    self.setup_var(input)
                    self.variables.add(input)
                self.add_client(input, (node, i))
            self.execute_callbacks("on_import", node, reason)

    def replace(
        self,
        pairs_to_replace: dict,
        reason: str = None,
        verbose: bool = None,
        import_missing: bool = False,
    ) -> None:
        """Replaces pairs of variables in the `FunctionGraph`.

        This is the main interface to manipulate the subgraph in `FunctionGraph`.
        For every node that uses `var` as input, makes it use `new_var` instead.

        Returns a dictionary with that maps old to new variables along with a
        list of all the changed input indices.
        memo = {old_variable: new_variable}

        Parameters
        ----------
        pairs_to_replacepairs_to_replace
            A dictionary in form of {var: new_var}
        reason
            The name of the optimization or operation in progress.
        verbose
            Print `reason`, `var`, and `new_var`.
        import_missing
            Import missing variables.

        """

        # TODO: Remove this if replace isn't supposed to be user facing
        assert isinstance(pairs_to_replace, dict), "Please switch to new interface"

        # Make a semi shallow copy of clients for later references
        old_clients = {}
        for var, new_var in list(pairs_to_replace.items()):
            if var == new_var:
                pairs_to_replace.pop(var)

        memo = pairs_to_replace.copy()
        for _var, _clients in self.clients.items():
            old_clients.update({_var: _clients.copy()})

        for var, new_var in pairs_to_replace.items():
            if verbose is None:
                verbose = config.optimizer_verbose
            if verbose:
                print(f"optimizer: rewrite {reason} replaces {var} with {new_var}")

            new_var = var.type.filter_variable(new_var, allow_convert=True)

            if not var.type == new_var.type:
                raise TypeError(
                    "The type of the replacement must be the"
                    " same as the type of the original Variable.",
                    var,
                    new_var,
                )

            if config.compute_test_value != "off":
                try:
                    tval = aesara.graph.op.get_test_value(var)
                    new_tval = aesara.graph.op.get_test_value(new_var)
                except TestValueError:
                    pass
                else:
                    tval_shape = getattr(tval, "shape", None)
                    new_tval_shape = getattr(new_tval, "shape", None)
                    if tval_shape != new_tval_shape:
                        raise AssertionError(
                            "The replacement variable has a test value with "
                            "a shape different from the original variable's "
                            f"test value. Original: {tval_shape}, new: {new_tval_shape}"
                        )

            if var not in self.variables:
                raise ValueError(
                    f"Given variable {var} is not a part of the FunctionGraph"
                )

            self.import_var(new_var, reason=reason, import_missing=import_missing)

            for node, i in old_clients[var]:
                assert node.inputs[i] == var
                # If an entry already exists in memo, that means the node has been
                # previously changed/cloned for changing some other input, hence the
                # new clone will be made using inputs of the previously cloned node
                # rather than original.
                curr_node = memo[node.out] if node in memo.keys() else node
                new_inputs = list(curr_node.inputs)
                # We make sure every input node is also updated according to memo,
                # since the input nodes other than ones specified may change.
                for _idx, _var in enumerate(new_inputs):
                    if _var in memo.keys():
                        new_inputs[_idx] = memo[_var]
                new_inputs[i] = new_var
                new_node = curr_node.op.make_node(*new_inputs)
                memo = self.create_next_nodes_memo(node, new_node, i, memo, old_clients)

        # By this time the memo will have all the mappings for the every replacements
        # that needs to happen in the FunctionGraph
        assert all(
            [
                isinstance(key, Variable) and isinstance(item, Variable)
                for key, item in memo.items()
            ]
        )

        unused_vars = self._replace_nodes(pairs_to_replace, old_clients, memo, reason)

        self.execute_callbacks(
            "on_replace_nodes",
            pairs_to_replace,
            unused_vars,
            old_clients,
            memo,
            reason=reason,
        )

        return memo, unused_vars

    def attach_feature(self, feature: Feature) -> None:
        """Add a ``graph.features.Feature`` to this function graph and trigger its ``on_attach`` callback."""
        # Filter out literally identical `Feature`s
        if feature in self._features:
            return  # the feature is already present

        # Filter out functionally identical `Feature`s.
        # `Feature`s may use their `on_attach` method to raise
        # `AlreadyThere` if they detect that some
        # installed `Feature` does the same thing already
        attach = getattr(feature, "on_attach", None)
        if attach is not None:
            try:
                attach(self)
            except AlreadyThere:
                return
        self.execute_callbacks_times.setdefault(feature, 0)
        # It would be nice if we could require a specific class instead of
        # a "workalike" so we could do actual error checking
        # if not isinstance(feature, Feature):
        #    raise TypeError("Expected Feature instance, got "+\
        #            str(type(feature)))

        # Add the feature
        self._features.append(feature)

    def remove_feature(self, feature: Feature) -> None:
        """Remove a feature from the graph.

        Calls ``feature.on_detach(function_graph)`` if an ``on_detach`` method
        is defined.

        """
        try:
            # Why do we catch the exception anyway?
            self._features.remove(feature)
        except ValueError:
            return
        detach = getattr(feature, "on_detach", None)
        if detach is not None:
            detach(self)

    def execute_callbacks(self, name: str, *args, **kwargs) -> None:
        """Execute callbacks.

        Calls ``getattr(feature, name)(*args)`` for each feature which has
        a method called after name.

        """
        t0 = time.time()
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                # this is safe because there is no work done inside the
                # try; the AttributeError really must come from feature.${name}
                # not existing
                continue
            tf0 = time.time()
            fn(self, *args, **kwargs)
            self.execute_callbacks_times[feature] += time.time() - tf0
        self.execute_callbacks_time += time.time() - t0

    def collect_callbacks(self, name: str, *args) -> Dict[Feature, Any]:
        """Collects callbacks

        Returns a dictionary d such that ``d[feature] == getattr(feature, name)(*args)``
        For each feature which has a method called after name.
        """
        d = {}
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            d[feature] = fn(*args)
        return d

    def toposort(self) -> List[Apply]:
        """Return a toposorted list of the nodes.

        Return an ordering of the graph's ``Apply`` nodes such that:

        * all the nodes of the inputs of a node are before that node and
        * they satisfy the orderings provided by each feature that has
          an ``orderings`` method.

        If a feature has an ``orderings`` method, it will be called with
        this `FunctionGraph` as sole argument. It should return a dictionary of
        ``{node: predecessors}`` where predecessors is a list of nodes that
        should be computed before the key node.
        """
        if len(self.apply_nodes) < 2:
            # optimization
            # when there are 0 or 1 nodes, no sorting is necessary
            # This special case happens a lot because the OpWiseCLinker
            # produces 1-element graphs.
            return list(self.apply_nodes)
        fg = self

        ords = self.orderings()

        order = io_toposort(fg.inputs, fg.outputs, ords)

        return order

    def orderings(self) -> Dict[Apply, List[Apply]]:
        """Return ``dict`` ``d`` s.t. ``d[node]`` is a list of nodes that must be evaluated before ``node`` itself can be evaluated.

        This is used primarily by the ``destroy_handler`` feature to ensure that
        the clients of any destroyed inputs have already computed their
        outputs.

        Notes
        -----
        This only calls the ``orderings()`` function on all features. It does not
        take care of computing the dependencies by itself.

        """
        assert isinstance(self._features, list)
        all_orderings = []

        for feature in self._features:
            if hasattr(feature, "orderings"):
                orderings = feature.orderings(self)
                if not isinstance(orderings, OrderedDict):
                    raise TypeError(
                        "Non-deterministic return value from "
                        + str(feature.orderings)
                        + ". Nondeterministic object is "
                        + str(orderings)
                    )
                if len(orderings) > 0:
                    all_orderings.append(orderings)
                    for node, prereqs in orderings.items():
                        if not isinstance(prereqs, (list, OrderedSet)):
                            raise TypeError(
                                "prereqs must be a type with a "
                                "deterministic iteration order, or toposort "
                                " will be non-deterministic."
                            )
        if len(all_orderings) == 1:
            # If there is only 1 ordering, we reuse it directly.
            return all_orderings[0].copy()
        else:
            # If there is more than 1 ordering, combine them.
            ords = OrderedDict()
            for orderings in all_orderings:
                for node, prereqs in orderings.items():
                    ords.setdefault(node, []).extend(prereqs)
            return ords

    def check_integrity(self) -> None:
        """Check the integrity of nodes in the graph."""
        nodes = set(applys_between(self.inputs, self.outputs))
        if self.apply_nodes != nodes:
            missing = nodes.difference(self.apply_nodes)
            excess = self.apply_nodes.difference(nodes)
            raise Exception(
                "The nodes are inappropriately cached. missing, in excess: ",
                missing,
                excess,
            )
        for node in nodes:
            for i, variable in enumerate(node.inputs):
                clients = self.clients[variable]
                if (node, i) not in clients:
                    raise Exception(
                        f"Inconsistent clients list {(node, i)} in {clients}"
                    )
        variables = set(vars_between(self.inputs, self.outputs))
        if set(self.variables) != variables:
            missing = variables.difference(self.variables)
            excess = self.variables.difference(variables)
            raise Exception(
                "The variables are inappropriately cached. missing, in excess: ",
                missing,
                excess,
            )
        for variable in variables:
            if (
                variable.owner is None
                and variable not in self.inputs
                and variable not in self.outputs
                and not isinstance(variable, Constant)
            ):
                raise Exception(f"Undeclared input: {variable}")
            for node, i in self.clients[variable]:
                if node not in nodes:
                    raise Exception(
                        f"Client not in FunctionGraph: {variable}, {(node, i)}"
                    )
                if node.inputs[i] is not variable:
                    raise Exception(
                        f"Inconsistent clients list: {variable}, {node.inputs[i]}"
                    )

    def __repr__(self):
        return f"FunctionGraph({', '.join(graph_as_string(self.inputs, self.outputs))})"

    def clone(self, check_integrity=True) -> "FunctionGraph":
        """Clone the graph."""
        return self.clone_get_equiv(check_integrity)[0]

    def clone_get_equiv(
        self, check_integrity: bool = True, attach_feature: bool = True
    ) -> Union["FunctionGraph", Dict[Variable, Variable]]:
        """Clone the graph and return a ``dict`` that maps old nodes to new nodes.

        Parameters
        ----------
        check_integrity
            Whether to check integrity.
        attach_feature
            Whether to attach feature of origin graph to cloned graph.

        Returns
        -------
        e
            Cloned fgraph. Every node in cloned graph is cloned.
        equiv
            A ``dict`` that maps old nodes to the new nodes.
        """
        equiv = clone_get_equiv(self.inputs, self.outputs)

        if check_integrity:
            self.check_integrity()
        e = FunctionGraph(
            [equiv[i] for i in self.inputs],
            [equiv[o] for o in self.outputs],
            clone=False,
        )
        if check_integrity:
            e.check_integrity()

        if attach_feature:
            for feature in self._features:
                e.attach_feature(feature)
        return e, equiv

    def __getstate__(self):
        # This is needed as some features introduce instance methods
        # This is not picklable
        d = self.__dict__.copy()
        for feature in self._features:
            for attr in getattr(feature, "pickle_rm_attr", []):
                del d[attr]
        # The class Updater take fct as parameter and they are lambda function, so unpicklable.

        # execute_callbacks_times have reference to optimizer, and they can't
        # be pickled as the decorators with parameters aren't pickable.
        if "execute_callbacks_times" in d:
            del d["execute_callbacks_times"]

        return d

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        for feature in self._features:
            if hasattr(feature, "unpickle"):
                feature.unpickle(self)

    def __contains__(self, item: Union[Variable, Apply]) -> bool:
        if isinstance(item, Variable):
            return item in self.variables
        elif isinstance(item, Apply):
            return item in self.apply_nodes
        else:
            raise TypeError()

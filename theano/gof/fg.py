"""
fg.py: fg stands for FunctionGraph
Contains the FunctionGraph class and exception
types that it can raise.

"""
import time
from collections import OrderedDict
from io import StringIO

import theano
from theano import config
from theano.gof import graph, toolbox, utils
from theano.gof.utils import TestValueError, get_variable_trace_string
from theano.misc.ordered_set import OrderedSet


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
        Exception.__init__(self, s)


class FunctionGraph(utils.object2):
    """
    A FunctionGraph represents a subgraph bound by a set of input variables and
    a set of output variables, ie a subgraph that specifies a theano function.
    The inputs list should contain all the inputs on which the outputs depend.
    Variables of type Constant are not counted as inputs.

    The FunctionGraph supports the replace operation which allows to replace a
    variable in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in theano.

    This class is also responsible for verifying that a graph is valid
    (ie, all the dtypes and broadcast patterns are compatible with the
    way the the Variables are used) and for annotating the Variables with
    a .clients field that specifies which Apply nodes use the variable.
    The .clients field combined with the .owner field and the Apply nodes'
    .inputs field allows the graph to be traversed in both directions.

    It can also be extended with new features using
    FunctionGraph.attach_feature(<toolbox.Feature instance>).
    See toolbox.Feature for event types and documentation.
    Extra features allow the FunctionGraph to verify new properties of
    a graph as it is optimized.
    # TODO: are there other things features can do to the fgraph?

    Historically, the FunctionGraph was called an Env. Keep this in mind
    while reading out-of-date documentation, e-mail support threads, etc.

    The constructor creates a FunctionGraph which operates on the subgraph
    bound by the inputs and outputs sets.

    This class keeps a pointer to the inputs and outputs, and also modifies
    them.

    #TODO: document what variables are[not] set in the FunctionGraph when a
    feature is added via the constructor. How constructed is the
    FunctionGraph?

    Parameters
    ----------
    inputs
        Inputs nodes of the graph, usually declared by the user.
    outputs
        Outputs nodes of the graph.
    clone
        If true, we will clone the graph. This is useful to remove the constant
        cache problem.

    Notes
    -----
    The intermediate nodes between 'inputs' and 'outputs' are not explicitely
    passed.

    """

    def __init__(self, inputs, outputs, features=None, clone=True, update_mapping=None):
        """
        Create an FunctionGraph which operates on the subgraph bound by the
        inputs and outputs sets.

        Parameters
        ----------
        inputs : list of theano.gof.graph.Variable
            Inputs nodes of the graph, usually declared by the user
        outputs : list of theano.gof.graph.Variable
            Outputs nodes of the graph.
        clone : boolean
            If true, we will clone the graph. This is useful to remove the
            constant cache problem.
        features : list of theano.gof.toolbox.Feature
            A list of features to be added to the `FunctionGraph`.
        update_mapping : dict
            Mapping between the inputs with updates and the outputs
            corresponding to their updates.
        """

        if clone:
            inputs, outputs = graph.clone(inputs, outputs)

        if not isinstance(inputs, list):
            raise TypeError("Argument `inputs` should be a list")

        if not isinstance(outputs, list):
            raise TypeError("Argument `outputs` should be a list")

        self.execute_callbacks_time = 0
        self.execute_callbacks_times = {}

        if features is None:
            features = []

        # XXX: Unless I'm missing something (but there's no documentation,
        # so I probably am) this should be a set.
        self._features = []

        # All apply nodes in the subgraph defined by inputs and
        # outputs are cached in this field
        self.apply_nodes = set()

        # Ditto for variable nodes.
        # It must contain all fgraph.inputs and all apply_nodes
        # outputs even if they aren't used in the graph.
        self.variables = set()

        # TODO FIXME: We should *not* be using a list created elsewhere!
        self.outputs = outputs

        for f in features:
            self.attach_feature(f)

        self.attach_feature(toolbox.ReplaceValidate())

        self.inputs = []
        for in_var in inputs:
            if in_var.owner is not None:
                raise ValueError(
                    "One of the provided inputs is the output of "
                    "an already existing node. "
                    "If that is okay, either discard that "
                    "input's owner or use graph.clone."
                )

            self.add_input(in_var, check=False)

        for output in outputs:
            self.import_var(output, reason="init")
        for i, output in enumerate(outputs):
            output.clients.append(("output", i))

        self.profile = None
        self.update_mapping = update_mapping

    def add_input(self, var, check=True):
        """Add a new variable as an input to this `FunctionGraph`.

        Parameters
        ----------
        var : theano.gof.graph.Variable

        """
        if check and var in self.inputs:
            return

        self.inputs.append(var)
        self.setup_var(var)
        self.variables.add(var)

    def setup_var(self, var):
        """Set up a variable so it belongs to this `FunctionGraph`.

        Parameters
        ----------
        var : theano.gof.graph.Variable

        """
        if hasattr(var, "fgraph") and var.fgraph is not None and var.fgraph is not self:
            raise Exception(f"{var} is already owned by another fgraph")
        var.fgraph = self
        var.clients = []
        # self.execute_callbacks('on_setup_variable', var)

    def setup_node(self, node):
        """Set up node so it belongs to this `FunctionGraph`.

        Parameters
        ----------
        node : theano.gof.graph.Apply

        """
        if hasattr(node, "fgraph") and node.fgraph is not self:
            raise Exception(f"{node} is already owned by another fgraph")
        if hasattr(node.op, "view_map") and not all(
            isinstance(view, (list, tuple)) for view in node.op.view_map.values()
        ):
            raise Exception(
                f"Op '{node.op}' have a bad view map '{node.op.view_map}',"
                " the values must be tuples or lists."
            )
        if hasattr(node.op, "destroy_map") and not all(
            isinstance(destroy, (list, tuple))
            for destroy in node.op.destroy_map.values()
        ):
            raise Exception(
                f"Op '{node.op}' have a bad destroy map '{node.op.destroy_map}',"
                " the values must be tuples or lists."
            )
        node.fgraph = self
        node.deps = {}
        # self.execute_callbacks('on_setup_node', node)

    def disown(self):
        """
        Cleans up all of this FunctionGraph's nodes and variables so they are
        not associated with this FunctionGraph anymore.

        The FunctionGraph should not be used anymore after disown is called.

        """
        for f in self._features:
            self.remove_feature(f)
        for apply_node in self.apply_nodes:
            del apply_node.fgraph
            del apply_node.deps
        for variable in self.variables:
            del variable.fgraph
            del variable.clients
        self.apply_nodes = set()
        self.variables = set()
        self.inputs = None
        self.outputs = None
        self.profile = None
        self.update_mapping = None

    def clients(self, var):
        """Return a list of all the `(node, i)` pairs such that `node.inputs[i]` is `var`.

        Told differently, a `list` of `(node, i)` such that each node have
        `var` as input at index `i`.

        """
        return var.clients

    def add_client(self, var, new_client):
        """Update the clients of `var` with `new_clients`.

        Parameters
        ----------
        var : Variable.
        new_client : (Apply, int)
            A `(node, i)` pair such that `node.inputs[i]` is `var`.

        """
        var.clients.append(new_client)

    def remove_client(self, var, client_to_remove, reason=None):
        """Recursively removes clients of a variable.

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
            A `(node, i)` pair such that `node.inputs[i]` will no longer be
            `var` in this `FunctionGraph`.

        """

        removal_stack = [(var, client_to_remove)]
        while removal_stack:
            var, client_to_remove = removal_stack.pop()

            try:
                var.clients.remove(client_to_remove)
            except ValueError:
                # In this case, the original `var` could've been removed from
                # the current `var`'s client list before this call.
                # There's nothing inherently wrong with that, so we continue as
                # if it were removed here.
                pass

            if var.clients:
                continue

            # Now, `var` has no more clients, so check if we need to remove it
            # and its `Apply` node
            if not var.owner:
                # The `var` is a `Constant` or an input without a client, so we
                # remove it
                self.variables.remove(var)

                # This allows us to quickly determine if `var` is still in the
                # `FunctionGraph`
                # TODO: It's a poor approach; remove it
                del var.fgraph
            else:
                apply_node = var.owner
                if not any(output.clients for output in apply_node.outputs):
                    # The `Apply` node is not used and is not an output, so we
                    # remove it and its outputs
                    if not hasattr(apply_node.tag, "removed_by"):
                        apply_node.tag.removed_by = []

                    apply_node.tag.removed_by.append(str(reason))

                    self.apply_nodes.remove(apply_node)

                    # del apply_node.fgraph
                    #
                    # for var in apply_node.outputs:
                    #     del var.fgraph

                    self.variables.difference_update(apply_node.outputs)

                    self.execute_callbacks("on_prune", apply_node, reason)

                    for i, in_var in enumerate(apply_node.inputs):
                        removal_stack.append((in_var, (apply_node, i)))

    def import_var(self, var, reason):
        """Import variables into this `FunctionGraph`.

        This will also import the `variable`'s `Apply` node.

        Parameters:
        ----------
        variable : theano.gof.graph.Variable
            The variable to be imported.
        reason : str
            The name of the optimization or operation in progress.
        """
        # Imports the owners of the variables
        if var.owner and var.owner not in self.apply_nodes:
            self.import_node(var.owner, reason=reason)
        elif (
            var.owner is None
            and not isinstance(var, graph.Constant)
            and var not in self.inputs
        ):
            from theano.gof.null_type import NullType

            if isinstance(var.type, NullType):
                raise TypeError(
                    "Computation graph contains a NaN. " + var.type.why_null
                )
            raise MissingInputError("Undeclared input", variable=var)
        if not getattr(var, "fgraph", None) is self:
            self.setup_var(var)
        self.variables.add(var)

    def import_node(self, apply_node, check=True, reason=None):
        """Recursively import everything between an `Apply` node and the `FunctionGraph`'s outputs.

        Parameters:
        ----------
        apply_node : theano.gof.graph.Apply
            The node to be imported.
        check : bool
            Check that the inputs for the imported nodes are also present in
            the `FunctionGraph`.
        reason : str
            The name of the optimization or operation in progress.
        """
        node = apply_node

        # We import the nodes in topological order. We only are interested
        # in new nodes, so we use all variables we know of as if they were the input set.
        # (the functions in the graph module only use the input set to
        # know where to stop going down)
        new_nodes = graph.io_toposort(self.variables, apply_node.outputs)

        if check:
            for node in new_nodes:
                if hasattr(node, "fgraph") and node.fgraph is not self:
                    raise Exception(f"{node} is already owned by another fgraph")
                for var in node.inputs:
                    if hasattr(var, "fgraph") and var.fgraph is not self:
                        raise Exception(f"{var} is already owned by another fgraph")
                    if (
                        var.owner is None
                        and not isinstance(var, graph.Constant)
                        and var not in self.inputs
                    ):
                        # Standard error message
                        error_msg = (
                            f"Input {int(node.inputs.index(var))} of the graph (indices start "
                            f"from 0), used to compute {node}, was not "
                            "provided and not given a value. Use the "
                            "Theano flag exception_verbosity='high', "
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
            assert node.fgraph is self
            self.execute_callbacks("on_import", node, reason)

    def change_input(self, node, i, new_var, reason=None):
        """Change ``node.inputs[i]`` to `new_var`.

        ``new_var.type == old_var.type`` must be ``True``, where ``old_var`` is the
        current value of ``node.inputs[i]`` which we want to replace.

        For each feature that has an `on_change_input` method, this method calls:
        ``feature.on_change_input(function_graph, node, i, old_var, new_var, reason)``

        Parameters
        ----------
        node : theano.gof.graph.Apply or str
            The node for which an input is to be changed.  If the value is
            the string ``"output"`` then the ``self.outputs`` will be used
            instead of ``node.inputs``.
        i : int
            The index in `node.inputs` that we want to change.
        new_var : theano.gof.graph.Variable
            The new variable to take the place of ``node.inputs[i]``.

        """
        # TODO: ERROR HANDLING FOR LISTENERS (should it complete the change or revert it?)
        if node == "output":
            r = self.outputs[i]
            if not r.type == new_var.type:
                raise TypeError(
                    "The type of the replacement must be the"
                    " same as the type of the original Variable.",
                    r,
                    new_var,
                )
            self.outputs[i] = new_var
        else:
            if node.fgraph is not self:
                raise Exception(
                    f"Cannot operate on {node} because it does not"
                    " belong to this FunctionGraph"
                )
            r = node.inputs[i]
            if not r.type == new_var.type:
                raise TypeError(
                    "The type of the replacement must be the"
                    " same as the type of the original Variable.",
                    r,
                    new_var,
                )
            node.inputs[i] = new_var

        if r is new_var:
            return

        self.import_var(new_var, reason=reason)
        self.add_client(new_var, (node, i))
        self.remove_client(r, (node, i), reason=reason)
        # Precondition: the substitution is semantically valid
        # However it may introduce cycles to the graph,  in which case the
        # transaction will be reverted later.
        self.execute_callbacks("on_change_input", node, i, r, new_var, reason=reason)

    def replace(self, var, new_var, reason=None, verbose=None):
        """Replace a variable in the `FunctionGraph`.

        This is the main interface to manipulate the subgraph in `FunctionGraph`.
        For every node that uses `var` as input, makes it use `new_var` instead.

        Parameters:
        ----------
        var : theano.gof.graph.Variable
            The variable to be replaced.
        new_var : theano.gof.graph.Variable
            The variable to replace `var`.
        reason : str
            The name of the optimization or operation in progress.
        verbose : bool
            Print `reason`, `var`, and `new_var`.

        """
        if verbose is None:
            verbose = config.optimizer_verbose
        if verbose:
            print(reason, var, new_var)

        if hasattr(var, "fgraph") and var.fgraph is not self:
            raise Exception(
                f"Cannot replace {var} because it does not belong to this FunctionGraph"
            )
        if var.type != new_var.type:
            new_var_2 = var.type.convert_variable(new_var)
            # We still make sure that the type converts correctly
            if new_var_2 is None or new_var_2.type != var.type:
                done = dict()
                used_ids = dict()
                old = theano.compile.debugmode.debugprint(
                    var,
                    prefix="  ",
                    depth=6,
                    file=StringIO(),
                    done=done,
                    print_type=True,
                    used_ids=used_ids,
                ).getvalue()
                new = theano.compile.debugmode.debugprint(
                    new_var,
                    prefix="  ",
                    depth=6,
                    file=StringIO(),
                    done=done,
                    print_type=True,
                    used_ids=used_ids,
                ).getvalue()
                raise toolbox.BadOptimization(
                    var,
                    new_var,
                    None,
                    None,
                    str(reason) + ". The type of the replacement must be the same.",
                    old,
                    new,
                )
            new_var = new_var_2

        if var not in self.variables:
            # this variable isn't in the graph... don't raise an
            # exception here, just return silently because it makes it
            # easier to implement some optimizations for
            # multiple-output ops
            return

        if theano.config.compute_test_value != "off":
            try:
                tval = theano.gof.op.get_test_value(var)
                new_tval = theano.gof.op.get_test_value(new_var)
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

        for node, i in list(var.clients):  # copy the client list for iteration
            assert (node == "output" and self.outputs[i] is var) or (
                node.inputs[i] is var
            )
            self.change_input(node, i, new_var, reason=reason)

    def replace_all(self, pairs, reason=None):
        """Replace variables in the `FunctionGraph` according to `(var, new_var)` pairs in a list."""
        for var, new_var in pairs:
            self.replace(var, new_var, reason=reason)

    def attach_feature(self, feature):
        """
        Adds a gof.toolbox.Feature to this function_graph and triggers its
        on_attach callback.

        """
        # Filter out literally identical features
        if feature in self._features:
            return  # the feature is already present

        # Filter out functionally identical features.
        # Features may use their on_attach method to raise
        # toolbox.AlreadyThere if they detect that some
        # installed feature does the same thing already
        attach = getattr(feature, "on_attach", None)
        if attach is not None:
            try:
                attach(self)
            except toolbox.AlreadyThere:
                return
        self.execute_callbacks_times.setdefault(feature, 0)
        # it would be nice if we could require a specific class instead of
        # a "workalike" so we could do actual error checking
        # if not isinstance(feature, toolbox.Feature):
        #    raise TypeError("Expected gof.toolbox.Feature instance, got "+\
        #            str(type(feature)))

        # Add the feature
        self._features.append(feature)

    def remove_feature(self, feature):
        """
        Removes the feature from the graph.

        Calls feature.on_detach(function_graph) if an on_detach method
        is defined.

        """
        try:
            # Why do we catch the exeception anyway?
            self._features.remove(feature)
        except ValueError:
            return
        detach = getattr(feature, "on_detach", None)
        if detach is not None:
            detach(self)

    def execute_callbacks(self, name, *args, **kwargs):
        """Execute callbacks

        Calls `getattr(feature, name)(*args)` for each feature which has
        a method called after name.

        """
        t0 = time.time()
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                # this is safe because there is no work done inside the
                # try; the AttributeError reall must come from feature.${name}
                # not existing
                continue
            tf0 = time.time()
            fn(self, *args, **kwargs)
            self.execute_callbacks_times[feature] += time.time() - tf0
        self.execute_callbacks_time += time.time() - t0

    def collect_callbacks(self, name, *args):
        """Collects callbacks

        Returns a dictionary d such that
        `d[feature] == getattr(feature, name)(*args)`
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

    def toposort(self):
        """Toposort

        Return an ordering of the graph's Apply nodes such that

        * All the nodes of the inputs of a node are before that node.
        * Satisfies the orderings provided by each feature that has
          an 'orderings' method.

        If a feature has an 'orderings' method, it will be called with
        this FunctionGraph as sole argument. It should return a dictionary of
        `{node: predecessors}` where predecessors is a list of nodes that
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

        order = graph.io_toposort(fg.inputs, fg.outputs, ords)

        return order

    def orderings(self):
        """Return `dict` `d` s.t. `d[node]` is a list of nodes that must be evaluated before `node` itself can be evaluated.

        This is used primarily by the destroy_handler feature to ensure that
        the clients of any destroyed inputs have already computed their
        outputs.

        Notes
        -----
        This only calls the `orderings()` function on all features. It does not
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

    def check_integrity(self):
        """
        Call this for a diagnosis if things go awry.

        """
        nodes = graph.ops(self.inputs, self.outputs)
        if self.apply_nodes != nodes:
            missing = nodes.difference(self.apply_nodes)
            excess = self.apply_nodes.difference(nodes)
            raise Exception(
                "The nodes are inappropriately cached. missing, in excess: ",
                missing,
                excess,
            )
        for node in nodes:
            if node.fgraph is not self:
                raise Exception("Node should belong to the FunctionGraph.", node)
            for i, variable in enumerate(node.inputs):
                if variable.fgraph is not self:
                    raise Exception(
                        "Input of node should belong to the FunctionGraph.",
                        variable,
                        (node, i),
                    )
                if (node, i) not in variable.clients:
                    raise Exception(
                        "Inconsistent clients list.", (node, i), variable.clients
                    )
        variables = set(graph.variables(self.inputs, self.outputs))
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
                and not isinstance(variable, graph.Constant)
            ):
                raise Exception("Undeclared input.", variable)
            if variable.fgraph is not self:
                raise Exception(
                    "Variable should belong to the FunctionGraph.", variable
                )
            for node, i in variable.clients:
                if node == "output":
                    if self.outputs[i] is not variable:
                        raise Exception(
                            "Inconsistent clients list.", variable, self.outputs[i]
                        )
                    continue
                if node not in nodes:
                    raise Exception("Client not in FunctionGraph.", variable, (node, i))
                if node.inputs[i] is not variable:
                    raise Exception(
                        "Inconsistent clients list.", variable, node.inputs[i]
                    )

    def __str__(self):
        return f"[{', '.join(graph.as_string(self.inputs, self.outputs))}]"

    def __repr__(self):
        return self.__str__()

    def clone(self, check_integrity=True):
        """
        Clone the graph and get a memo( a dict )that map old node to new node

        """
        return self.clone_get_equiv(check_integrity)[0]

    def clone_get_equiv(self, check_integrity=True, attach_feature=True):
        """Clone the graph and get a dict that maps old nodes to new ones

        Parameters:
            check_integrity: bool
                Whether to check integrity. Default is True.
            attach_feature: bool
                Whether to attach feature of origin graph to cloned graph.
                Default is True.

        Returns:
            e: FunctionGraph
                Cloned fgraph. Every node in cloned graph is cloned.
            equiv: dict
                A dict that map old node to new node.
        """
        equiv = graph.clone_get_equiv(self.inputs, self.outputs)

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
        """
        This is needed as some features introduce instance methods.
        This is not picklable.

        """
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

from collections.abc import Sequence
from warnings import warn

from theano.graph.basic import Constant
from theano.link.basic import Container, PerformLinker
from theano.link.utils import gc_helper, map_storage, streamline
from theano.utils import difference


class JAXLinker(PerformLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX.

    Attributes
    ----------
    allow_non_jax: bool
        A boolean indicating whether or not an exception is thrown when the
        graph cannot be JAX compiled (e.g. the graph has an unsupported operator).
        If `allow_non_jax` is `True`, the fallback is currently Python compilation.

    """

    allow_non_jax = False

    def create_jax_thunks(self, compute_map, storage_map):
        """Create a thunk for each output of the `Linker`s `FunctionGraph`.

        This is differs from the other thunk-making function in that it only
        produces thunks for the `FunctionGraph` output nodes.

        Parameters
        ----------
        compute_map: dict
            The compute map dictionary.
        storage_map: dict
            The storage map dictionary.

        Returns
        -------
        thunks: list
            A tuple containing the thunks.
        output_nodes: list and their
            A tuple containing the output nodes.

        """
        import jax

        from theano.link.jax.jax_dispatch import jax_funcify

        output_nodes = [o.owner for o in self.fgraph.outputs]

        # Create a JAX-compilable function from our `FunctionGraph`
        jaxed_fgraph_outputs = jax_funcify(self.fgraph)

        assert len(jaxed_fgraph_outputs) == len(output_nodes)

        # I suppose we can consider `Constant`s to be "static" according to
        # JAX.
        static_argnums = [
            n for n, i in enumerate(self.fgraph.inputs) if isinstance(i, Constant)
        ]

        thunk_inputs = [storage_map[n] for n in self.fgraph.inputs]

        thunks = []

        for node, jax_funcs in zip(output_nodes, jaxed_fgraph_outputs):

            thunk_outputs = [storage_map[n] for n in node.outputs]

            if not isinstance(jax_funcs, Sequence):
                jax_funcs = [jax_funcs]

            jax_impl_jits = [
                jax.jit(jax_func, static_argnums) for jax_func in jax_funcs
            ]

            def thunk(
                node=node, jax_impl_jits=jax_impl_jits, thunk_outputs=thunk_outputs
            ):
                outputs = [
                    jax_impl_jit(*[x[0] for x in thunk_inputs])
                    for jax_impl_jit in jax_impl_jits
                ]

                if len(jax_impl_jits) < len(node.outputs):
                    # In this case, the JAX function will output a single
                    # output that contains the other outputs.
                    # This happens for multi-output `Op`s that directly
                    # correspond to multi-output JAX functions (e.g. `SVD` and
                    # `jax.numpy.linalg.svd`).
                    outputs = outputs[0]

                for o_node, o_storage, o_val in zip(
                    node.outputs, thunk_outputs, outputs
                ):
                    compute_map[o_node][0] = True
                    if len(o_storage) > 1:
                        assert len(o_storage) == len(o_val)
                        for i, o_sub_val in enumerate(o_val):
                            o_storage[i] = o_sub_val
                    else:
                        o_storage[0] = o_val
                return outputs

            thunk.inputs = thunk_inputs
            thunk.outputs = thunk_outputs
            thunk.lazy = False

            thunks.append(thunk)

        return thunks, output_nodes

    def make_all(self, input_storage=None, output_storage=None, storage_map=None):
        fgraph = self.fgraph
        nodes = self.schedule(fgraph)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = map_storage(
            fgraph, nodes, input_storage, output_storage, storage_map
        )

        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        try:
            # We need to create thunk functions that will populate the output
            # storage arrays with the JAX-computed values.
            thunks, nodes = self.create_jax_thunks(compute_map, storage_map)

        except NotImplementedError as e:
            if not self.allow_non_jax:
                raise

            warn(f"JaxLinker could not JAXify graph: {e}")

            thunks = []
            for node in nodes:
                thunk = node.op.make_thunk(
                    node, storage_map, compute_map, no_recycling, "py"
                )
                thunk_inputs = [storage_map[v] for v in node.inputs]
                thunk_outputs = [storage_map[v] for v in node.outputs]

                thunk.inputs = thunk_inputs
                thunk.outputs = thunk_outputs

                thunks.append(thunk)

        computed, last_user = gc_helper(nodes)

        if self.allow_gc:
            post_thunk_old_storage = []

            for node in nodes:
                post_thunk_old_storage.append(
                    [
                        storage_map[input]
                        for input in node.inputs
                        if (input in computed)
                        and (input not in fgraph.outputs)
                        and (node == last_user[input])
                    ]
                )
        else:
            post_thunk_old_storage = None

        if no_recycling is True:
            no_recycling = list(storage_map.values())
            no_recycling = difference(no_recycling, input_storage)
        else:
            no_recycling = [
                storage_map[r] for r in no_recycling if r not in fgraph.inputs
            ]

        fn = streamline(
            fgraph, thunks, nodes, post_thunk_old_storage, no_recycling=no_recycling
        )

        fn.allow_gc = self.allow_gc
        fn.storage_map = storage_map

        return (
            fn,
            [
                Container(input, storage)
                for input, storage in zip(fgraph.inputs, input_storage)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage)
            ],
            thunks,
            nodes,
        )

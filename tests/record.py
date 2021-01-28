from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.link.basic import WrapLinkerMany
from aesara.link.vm import VMLinker
from aesara.printing import hex_digest, min_informative_str


__authors__ = ["PyMC Team", "Ian Goodfellow"]


class MismatchError(Exception):
    """
    Raised by Record.handle_line when the
    current execution doesn't match the replay
    of a record.
    """


class Record:
    """
    Records a sequence of strings (from a string buffer). These can then be
    compared to another sequence of strings, and if the two sequences don't
    match a mismatch exception is raised.

    Example:
       # Create a Record object and store 'hello world' inside it
       output = cStringIO.StringIO()
       recorder = Record(file_object=output, replay=False)
       recorder.handle_line('hello world \n')

       # Store the previous output
       output_value = output.getvalue()
       output = cStringIO.StringIO(output_value)

       # Create another Record object, now in playback mode, and set
       # it to the previous sequence of strings
       playback_checker = Record(file_object=output,  replay=True)

       # Check if it matches the previous one
       playback_checker.handle_line('hello world \n')

       # Now check if it the next item matches something else. This will
       # throw an exception because there is no next item
       playback_checker.handle_line('hello new world \n')
    """

    def __init__(self, file_object=None, file_path=None, replay=False):
        """
        Initializes Record object to use file on disc and whether it is in
        replay mode or not.

        Parameters
        ----------
        file_object : StringIO
            The input string buffer.
        file_path : string, optional
            File to save Record to.
        replay : bool, optional
            Determines whether or not the object is in playback mode. If not
            in playback mode, the content of record will be written to the
            file. If in playback mode, the content of file is loaded into the
            record.
        """

        assert file_object is not None or file_path is not None

        if replay and file_object is None:
            self.f = open(file_path)
        elif (not replay) and file_object is None:
            self.f = open(file_path, "w")
        else:
            self.f = file_object

        self.__dict__.update(locals())

    def handle_line(self, line):
        """
        If not in playback mode, it records a new string. If in playback mode,
        it compares the current string to the next element in the sequence.
        If these are identical the element is removed and otherwise a mismatch
        exception is raised.

        Parameters
        ----------
        line : string
            The string to record.
        """

        assert line.endswith("\n")
        assert line[:-2].find("\n") == -1
        if self.replay:
            old_line = self.f.readline()
            if old_line != line:
                msg = "Replay detected mismatch.\n"
                msg += " I wanted to write:\n"
                if len(line) > 100:
                    msg += line[0:100] + "..."
                else:
                    msg += line
                msg += "\nwhen previous job wrote:\n"
                if len(old_line) > 100:
                    msg += old_line[0:100] + "..."
                else:
                    msg += old_line
                raise MismatchError(msg)
        else:
            self.f.write(line)


class RecordMode(Mode):
    """
    Records all computations done with a function in a file at output_path.
    Writes into the file the index of each apply node and sha256 digests of the
    numpy ndarrays it receives as inputs and produces as output.

    Example:
       # We use RecordMode to test that the computation of a function is
       identical. Create a Record object and use it to initialize a
       RecordMode object.
       output = cStringIO.StringIO()
       record = Record(file_object=output, replay=False)
       record_mode = RecordMode(record)

       # Then compile and call the function you wish to test, which uses
       # Apply nodes with record_mode as first parameter to record all the
       # computations to file. For example, call an Aesara function with the
       # RecordMode object.
       x = aesara.tensor.dscalar()
       f = aesara.function([x], 2*x, mode=record_mode)
       print f(4)

       # Create another RecordMode object and initialize it with the previous
       # record.
       output = cStringIO.StringIO(output.getvalue())
       playback = Record(file_object=output, replay=True)
       playback_mode = RecordMode(playback)

       # Compile and call the function to test again with record_mode as
       # first parameter. An exception will be thrown if the recorded
       # computations are not identical between the two runs.
       x = aesara.tensor.dscalar()
       f = aesara.function([x], 2*x, mode=playback_mode)
       print f(4)

    """

    def set_record(self, record):
        """
        Configure object to use an existing Record object.

        Parameters
        ----------
        record : Record
            The Record object to use.
        """

        self.record = record
        self.known_fgraphs = set()

    def __init__(self, record=None, **kwargs):
        """
        Takes either a Record object or the keyword arguments to make one.

        Parameters
        ----------
        record : Record
            The existing Record object to use.
        kwargs : pointer?
            Keyword arguments to construct new object.
        """

        if record is None:
            record = Record(**kwargs)
        else:
            assert len(kwargs.keys()) == 0

        self.set_record(record)

        def handle_line(fgraph, line, i, node, fn):
            """
            Records new node computation.

            Parameters
            ----------
            line : string
                Line to record. For example, the function name or node name.
            i : integer
                Node number in the toposort order.
            node : Apply,
                The Apply node which created the entry.
            fn : Function,
                Function related to Apply node.
            """
            try:
                self.record.handle_line(line)
            except MismatchError as e:
                print("Got this MismatchError:")
                print(e)
                print(f"while processing node i={i}:")
                print(f"str(node):{node}")
                print("Symbolic inputs: ")
                for elem in node.inputs:
                    print(min_informative_str(elem))
                print("str(output) of outputs: ")
                for elem in fn.outputs:
                    assert isinstance(elem, list)
                    (elem,) = elem
                    print(str(elem))
                print(f"function name: {fgraph.name}")
                raise MismatchError("Non-determinism detected by WrapLinker")

        def callback(fgraph, i, node, fn):
            """
            Function called by Apply nodes at the end of each computation?
            """

            if fgraph.name is None:
                raise ValueError(
                    "Un-named functions are not allowed with RecordMode, "
                    "because they make it impossible to tell if the same function is "
                    "running during the playback."
                )

            if fgraph not in self.known_fgraphs:
                assert not any(
                    [elem.name == fgraph.name for elem in self.known_fgraphs]
                )
                self.known_fgraphs.add(fgraph)
                num_app = len(fgraph.apply_nodes)
                line = f"Function {fgraph.name} has {num_app} apply nodes.\n"
                handle_line(fgraph, line, i, node, fn)

            line = f"Function name: {fgraph.name}\n"
            handle_line(fgraph, line, i, node, fn)
            line = f"Node {i}:{node}\n"
            handle_line(fgraph, line, i, node, fn)
            assert all([isinstance(x, list) and len(x) == 1 for x in fn.inputs])

            def digest(x):
                x = x[0]
                return hex_digest(x)

            inputs_digest = " ".join([digest(x) for x in fn.inputs])
            line = f"Inputs: {inputs_digest}\n"
            handle_line(fgraph, line, i, node, fn)
            fn()
            outputs_digest = " ".join([digest(x) for x in fn.outputs])
            line = f"Outputs: {outputs_digest}\n"
            handle_line(fgraph, line, i, node, fn)

        # linker = aesara.link.c.basic.OpWiseCLinker()
        linker = VMLinker(use_cloop=bool(config.cxx))

        wrap_linker = WrapLinkerMany([linker], [callback])
        super().__init__(wrap_linker, optimizer="fast_run")

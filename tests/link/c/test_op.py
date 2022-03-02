import numpy as np
import pytest

import aesara
from aesara import scalar as aes
from aesara.configdefaults import config
from aesara.graph.basic import Apply
from aesara.graph.utils import MethodNotDefined
from aesara.link.c.op import COp


class StructOp(COp):
    __props__ = ()

    def do_constant_folding(self, fgraph, node):
        # we are not constant
        return False

    # The input only serves to distinguish thunks
    def make_node(self, i):
        return Apply(self, [i], [aes.uint64()])

    def c_support_code_struct(self, node, name):
        return f"npy_uint64 counter{name};"

    def c_init_code_struct(self, node, name, sub):
        return f"counter{name} = 0;"

    def c_code(self, node, name, input_names, outputs_names, sub):
        return """
%(out)s = counter%(name)s;
counter%(name)s++;
""" % dict(
            out=outputs_names[0], name=name
        )

    def c_code_cache_version(self):
        return (1,)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")


class TestCOp:
    @pytest.mark.skipif(
        not config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_op_struct(self):
        sop = StructOp()
        c = sop(aesara.tensor.constant(0))
        mode = None
        if config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        f = aesara.function([], c, mode=mode)
        rval = f()
        assert rval == 0
        rval = f()
        assert rval == 1

        c2 = sop(aesara.tensor.constant(1))
        f2 = aesara.function([], [c, c2], mode=mode)
        rval = f2()
        assert rval == [0, 0]


class TestMakeThunk:
    def test_no_c_code(self):
        class IncOnePython(COp):
            """An Op with only a Python (perform) implementation"""

            __props__ = ()

            def make_node(self, input):
                input = aes.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def perform(self, node, inputs, outputs):
                (input,) = inputs
                (output,) = outputs
                output[0] = input + 1

        i = aes.int32("i")
        o = IncOnePython()(i)

        # Check that the c_code function is not implemented
        with pytest.raises(NotImplementedError):
            o.owner.op.c_code(o.owner, "o", ["x"], "z", {"fail": ""})

        storage_map = {i: [np.int32(3)], o: [None]}
        compute_map = {i: [True], o: [False]}

        thunk = o.owner.op.make_thunk(
            o.owner, storage_map, compute_map, no_recycling=[]
        )

        required = thunk()
        # Check everything went OK
        assert not required  # We provided all inputs
        assert compute_map[o][0]
        assert storage_map[o][0] == 4

    def test_no_perform(self):
        class IncOneC(COp):
            """An Op with only a C (c_code) implementation"""

            __props__ = ()

            def make_node(self, input):
                input = aes.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def c_code(self, node, name, inputs, outputs, sub):
                (x,) = inputs
                (z,) = outputs
                return f"{z} = {x} + 1;"

            def perform(self, *args, **kwargs):
                raise NotImplementedError("No Python implementation available.")

        i = aes.int32("i")
        o = IncOneC()(i)

        # Check that the perform function is not implemented
        with pytest.raises((NotImplementedError, MethodNotDefined)):
            o.owner.op.perform(o.owner, 0, [None])

        storage_map = {i: [np.int32(3)], o: [None]}
        compute_map = {i: [True], o: [False]}

        thunk = o.owner.op.make_thunk(
            o.owner, storage_map, compute_map, no_recycling=[]
        )
        if config.cxx:
            required = thunk()
            # Check everything went OK
            assert not required  # We provided all inputs
            assert compute_map[o][0]
            assert storage_map[o][0] == 4
        else:
            with pytest.raises((NotImplementedError, MethodNotDefined)):
                thunk()

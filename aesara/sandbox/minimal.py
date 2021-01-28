import numpy as np

from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor.type import lscalar


class Minimal(Op):
    # TODO : need description for class

    # if the Op has any attributes, consider using them in the eq function.
    # If two Apply nodes have the same inputs and the ops compare equal...
    # then they will be MERGED so they had better have computed the same thing!

    __props__ = ()

    def __init__(self):
        # If you put things here, think about whether they change the outputs
        # computed by # self.perform()
        #  - If they do, then you should take them into consideration in
        #    __eq__ and __hash__
        #  - If they do not, then you should not use them in
        #    __eq__ and __hash__

        super().__init__()

    def make_node(self, *args):
        # HERE `args` must be AESARA VARIABLES
        return Apply(op=self, inputs=args, outputs=[lscalar()])

    def perform(self, node, inputs, out_):
        (output,) = out_
        # HERE `inputs` are PYTHON OBJECTS

        # do what you want here,
        # but do not modify any of the arguments [inplace].
        print("perform got %i arguments" % len(inputs))

        print("Max of input[0] is ", np.max(inputs[0]))

        # return some computed value.
        # do not return something that is aliased to one of the inputs.
        output[0] = np.asarray(0, dtype="int64")


minimal = Minimal()

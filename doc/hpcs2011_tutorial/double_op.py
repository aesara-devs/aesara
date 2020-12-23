import numpy as np

import theano
from theano.gof.graph import Apply
from theano.gof.op import Op


class DoubleOp(Op):
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * 2


x = theano.tensor.matrix()

f = theano.function([x], DoubleOp()(x))

inp = np.random.rand(5, 5)
out = f(inp)
assert np.allclose(inp * 2, out)
print(inp)
print(out)

import numpy as np

import aesara


class DoubleOp(aesara.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = aesara.tensor.as_tensor_variable(x)
        return aesara.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * 2


x = aesara.tensor.matrix()

f = aesara.function([x], DoubleOp()(x))

inp = np.random.rand(5, 5)
out = f(inp)
assert np.allclose(inp * 2, out)
print(inp)
print(out)

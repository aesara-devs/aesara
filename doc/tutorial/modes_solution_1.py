#!/usr/bin/env python
# Aesara tutorial
# Solution to Exercise in section 'Configuration Settings and Compiling Modes'

import numpy as np
import aesara
import aesara.tensor as at


aesara.config.floatX = "float32"

rng = np.random.default_rng(428)

N = 400
feats = 784
D = (
    rng.standard_normal((N, feats)).astype(aesara.config.floatX),
    rng.integers(size=N, low=0, high=2).astype(aesara.config.floatX),
)
training_steps = 10000

# Declare Aesara symbolic variables
x = at.matrix("x")
y = at.vector("y")
w = aesara.shared(rng.standard_normal(feats).astype(aesara.config.floatX), name="w")
b = aesara.shared(np.asarray(0.0, dtype=aesara.config.floatX), name="b")
x.tag.test_value = D[0]
y.tag.test_value = D[1]
# print "Initial model:"
# print w.get_value(), b.get_value()

# Construct Aesara expression graph
p_1 = 1 / (1 + at.exp(-at.dot(x, w) - b))  # Probability of having a one
prediction = p_1 > 0.5  # The prediction that is done: 0 or 1
xent = -y * at.log(p_1) - (1 - y) * at.log(1 - p_1)  # Cross-entropy
cost = at.cast(xent.mean(), "float32") + 0.01 * (w**2).sum()  # The cost to optimize
gw, gb = at.grad(cost, [w, b])

# Compile expressions to functions
train = aesara.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates={w: w - 0.01 * gw, b: b - 0.01 * gb},
    name="train",
)
predict = aesara.function(inputs=[x], outputs=prediction, name="predict")

if any(
    x.op.__class__.__name__ in ("Gemv", "CGemv", "Gemm", "CGemm")
    for x in train.maker.fgraph.toposort()
):
    print("Used the cpu")
else:
    print("ERROR, not able to tell if aesara used the cpu or another device")
    print(train.maker.fgraph.toposort())

for i in range(training_steps):
    pred, err = train(D[0], D[1])
# print "Final model:"
# print w.get_value(), b.get_value()

print("target values for D")
print(D[1])

print("prediction on D")
print(predict(D[0]))

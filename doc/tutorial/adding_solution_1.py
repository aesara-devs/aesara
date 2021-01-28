#!/usr/bin/env python
# Aesara tutorial
# Solution to Exercise in section 'Baby Steps - Algebra'


import aesara
a = aesara.tensor.vector()  # declare variable
b = aesara.tensor.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = aesara.function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]

import sys


print(
    "DEPRECATION: theano.sandbox.conv no longer provides conv. "
    "They have been moved to theano.tensor.nnet.conv",
    file=sys.stderr,
)

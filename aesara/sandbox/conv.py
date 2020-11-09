import sys


print(
    "DEPRECATION: aesara.sandbox.conv no longer provides conv. "
    "They have been moved to aesara.tensor.nnet.conv",
    file=sys.stderr,
)

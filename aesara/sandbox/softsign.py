from aesara.tensor.nnet.nnet import softsign  # noqa
import sys


print(
    "DEPRECATION WARNING: softsign was moved from aesara.sandbox.softsign to "
    "aesara.tensor.nnet.nnet ",
    file=sys.stderr,
)

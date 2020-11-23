from theano.tensor.nnet.nnet import softsign  # noqa
from warnings import warn


warn(
    "softsign was moved from theano.sandbox.softsign to theano.tensor.nnet.nnet ",
    category=DeprecationWarning,
)

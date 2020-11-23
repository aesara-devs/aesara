from warnings import warn


warn(
    "theano.sandbox.conv no longer provides conv. "
    "They have been moved to theano.tensor.nnet.conv",
    category=DeprecationWarning,
)

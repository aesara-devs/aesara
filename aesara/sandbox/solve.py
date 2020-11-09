import warnings


from aesara.tensor.slinalg import solve  # noqa

message = (
    "The module aesara.sandbox.solve will soon be deprecated.\n"
    "Please use tensor.slinalg.solve instead."
)

warnings.warn(message)

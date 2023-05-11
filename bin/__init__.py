import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
import warnings

warnings.warn(
    message= "Importing 'bin.aesara_cache' is deprecated. Import from "
    "'aesara.bin.aesara_cache' instead.",
    category=DeprecationWarning,
    stacklevel=2,  # Raise the warning on the import line
)

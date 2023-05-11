import warnings
warnings.warn("Please replace 'aesara.tensor.sub' with 'aesara.tensor.subtract'.", DeprecationWarning)
#!/usr/bin/env python

import warnings

from aesara.bin.aesara_cache import *
from aesara.bin.aesara_cache import _logger

if __name__ == "__main__":
    warnings.warn(
        message= "Running 'aesara_cache.py' is deprecated. Use the aesara-cache "
        "script instead.",
        category=DeprecationWarning,
    )
    main()

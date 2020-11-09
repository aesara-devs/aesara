import pytest

import aesara.gpuarray
import aesara.tensor


if aesara.gpuarray.pygpu is None:
    pytest.skip("pygpu not installed", allow_module_level=True)


init_error = None
if not aesara.gpuarray.pygpu_activated and not aesara.config.force_device:
    try:
        aesara.gpuarray.init_dev("cuda")
    except Exception as e:
        init_error = e

if not aesara.gpuarray.pygpu_activated:
    if init_error:
        pytest.skip(str(init_error), allow_module_level=True)
    else:
        pytest.skip("pygpu disabled", allow_module_level=True)

test_ctx_name = None

if aesara.config.mode == "FAST_COMPILE":
    mode_with_gpu = (
        aesara.compile.mode.get_mode("FAST_RUN").including("gpuarray").excluding("gpu")
    )
    mode_without_gpu = aesara.compile.mode.get_mode("FAST_RUN").excluding("gpuarray")
else:
    mode_with_gpu = (
        aesara.compile.mode.get_default_mode().including("gpuarray").excluding("gpu")
    )
    mode_without_gpu = aesara.compile.mode.get_default_mode().excluding("gpuarray")
    mode_without_gpu.check_py_code = False


# If using float16, cast reference input to float32
def ref_cast(x):
    if x.type.dtype == "float16":
        x = aesara.tensor.cast(x, "float32")
    return x

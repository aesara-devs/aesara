"""
Test config options.
"""
import logging
from unittest.mock import patch

from theano.configdefaults import default_blas_ldflags
from theano.configparser import THEANO_FLAGS_DICT, AddConfigVar, ConfigParam


def test_invalid_default():
    # Ensure an invalid default value found in the Theano code only causes
    # a crash if it is not overridden by the user.

    def filter(val):
        if val == "invalid":
            raise ValueError()
        else:
            return val

    try:
        # This should raise a ValueError because the default value is
        # invalid.
        AddConfigVar(
            "T_config.test_invalid_default_a",
            doc="unittest",
            configparam=ConfigParam("invalid", filter=filter),
            in_c_key=False,
        )
        raise AssertionError()
    except ValueError:
        pass

    THEANO_FLAGS_DICT["T_config.test_invalid_default_b"] = "ok"
    # This should succeed since we defined a proper value, even
    # though the default was invalid.
    AddConfigVar(
        "T_config.test_invalid_default_b",
        doc="unittest",
        configparam=ConfigParam("invalid", filter=filter),
        in_c_key=False,
    )

    # Check that the flag has been removed
    assert "T_config.test_invalid_default_b" not in THEANO_FLAGS_DICT

    # TODO We should remove these dummy options on test exit.


@patch("theano.configdefaults.try_blas_flag", return_value=None)
@patch("theano.configdefaults.sys")
def test_default_blas_ldflags(sys_mock, try_blas_flag_mock, caplog):

    sys_mock.version = "3.8.0 | packaged by conda-forge | (default, Nov 22 2019, 19:11:38) \n[GCC 7.3.0]"

    with patch.dict("sys.modules", {"mkl": None}):
        with caplog.at_level(logging.WARNING):
            default_blas_ldflags()

    assert "install mkl with" in caplog.text

"""Test config options."""
import logging
from unittest.mock import patch

import pytest

from theano import configparser
from theano.configdefaults import default_blas_ldflags
from theano.configparser import THEANO_FLAGS_DICT, AddConfigVar, ConfigParam


def test_invalid_default():
    # Ensure an invalid default value found in the Theano code only causes
    # a crash if it is not overridden by the user.

    def validate(val):
        if val == "invalid":
            raise ValueError("Test-triggered")

    with pytest.raises(ValueError, match="Test-triggered"):
        # This should raise a ValueError because the default value is
        # invalid.
        AddConfigVar(
            "T_config.test_invalid_default_a",
            doc="unittest",
            configparam=ConfigParam("invalid", validate=validate),
            in_c_key=False,
        )

    THEANO_FLAGS_DICT["T_config.test_invalid_default_b"] = "ok"
    # This should succeed since we defined a proper value, even
    # though the default was invalid.

    THEANO_FLAGS_DICT["T_config.test_invalid_default_b"] = "ok"
    # This should succeed since we defined a proper value, even
    # though the default was invalid.
    AddConfigVar(
        "T_config.test_invalid_default_b",
        doc="unittest",
        configparam=ConfigParam("invalid", validate=validate),
        in_c_key=False,
    )

    # TODO We should remove these dummy options on test exit.
    # Check that the flag has been removed
    assert "T_config.test_invalid_default_b" not in THEANO_FLAGS_DICT


@patch("theano.configdefaults.try_blas_flag", return_value=None)
@patch("theano.configdefaults.sys")
def test_default_blas_ldflags(sys_mock, try_blas_flag_mock, caplog):

    sys_mock.version = "3.8.0 | packaged by conda-forge | (default, Nov 22 2019, 19:11:38) \n[GCC 7.3.0]"

    with patch.dict("sys.modules", {"mkl": None}):
        with caplog.at_level(logging.WARNING):
            default_blas_ldflags()

    assert "install mkl with" in caplog.text


def test_config_param_apply_and_validation():
    cp = ConfigParam(
        "TheDeFauLt",
        apply=lambda v: v.lower(),
        validate=lambda v: v in "thedefault,thesetting",
        mutable=True,
    )
    assert cp.default == "TheDeFauLt"
    assert not hasattr(cp, "val")

    # can't assign invalid value
    with pytest.raises(ValueError, match="Invalid value"):
        cp.__set__("cls", "invalid")

    assert not hasattr(cp, "val")

    # effectivity of apply function
    cp.__set__("cls", "THESETTING")
    assert cp.val == "thesetting"

    # respect the mutability
    cp._mutable = False
    with pytest.raises(Exception, match="Can't change"):
        cp.__set__("cls", "THEDEFAULT")


def test_config_types_bool():
    valids = {
        True: ["1", 1, True, "true", "True"],
        False: ["0", 0, False, "false", "False"],
    }

    param = configparser.BoolParam(None)

    assert isinstance(param, configparser.ConfigParam)
    assert param.default is None

    for outcome, inputs in valids.items():
        for input in inputs:
            applied = param.apply(input)
            assert applied == outcome
            assert param.validate(applied) is not False

    with pytest.raises(ValueError, match="Invalid value"):
        param.apply("notabool")

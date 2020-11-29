"""Test config options."""
import logging
from unittest.mock import patch

import pytest

from theano import configdefaults, configparser
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
            "T_config__test_invalid_default_a",
            doc="unittest",
            configparam=ConfigParam("invalid", validate=validate),
            in_c_key=False,
        )

    THEANO_FLAGS_DICT["T_config__test_invalid_default_b"] = "ok"
    # This should succeed since we defined a proper value, even
    # though the default was invalid.
    AddConfigVar(
        "T_config__test_invalid_default_b",
        doc="unittest",
        configparam=ConfigParam("invalid", validate=validate),
        in_c_key=False,
    )

    # TODO We should remove these dummy options on test exit.
    # Check that the flag has been removed
    assert "T_config__test_invalid_default_b" not in THEANO_FLAGS_DICT


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


def test_config_hash():
    # TODO: use custom config instance for the test
    root = configparser.config
    configparser.AddConfigVar(
        "test_config_hash",
        "A config var from a test case.",
        configparser.StrParam("test_default"),
        root=root,
    )

    h0 = configparser.get_config_hash()

    with configparser.change_flags(test_config_hash="new_value"):
        assert root.test_config_hash == "new_value"
        h1 = configparser.get_config_hash()

    h2 = configparser.get_config_hash()
    assert h1 != h0
    assert h2 == h0


class TestConfigTypes:
    def test_bool(self):
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

    def test_enumstr(self):
        cp = configparser.EnumStr("blue", ["red", "green", "yellow"])
        assert len(cp.all) == 4
        with pytest.raises(ValueError, match=r"Invalid value \('foo'\)"):
            cp.apply("foo")
        with pytest.raises(ValueError, match="Non-str value"):
            configparser.EnumStr(default="red", options=["red", 12, "yellow"])

    def test_deviceparam(self):
        cp = configparser.DeviceParam("cpu", mutable=False)
        assert cp.default == "cpu"
        assert cp._apply("cuda123") == "cuda123"
        with pytest.raises(ValueError, match="old GPU back-end"):
            cp._apply("gpu123")
        with pytest.raises(ValueError, match="Invalid value"):
            cp._apply("notadevice")
        assert str(cp) == "None (cpu, opencl*, cuda*) "


def test_config_context():
    # TODO: use custom config instance for the test
    root = configparser.config
    configparser.AddConfigVar(
        "test_config_context",
        "A config var from a test case.",
        configparser.StrParam("test_default"),
        root=root,
    )
    assert hasattr(root, "test_config_context")
    assert root.test_config_context == "test_default"

    with configparser.change_flags(test_config_context="new_value"):
        assert root.test_config_context == "new_value"
    assert root.test_config_context == "test_default"


def test_no_more_dotting():
    with pytest.raises(ValueError, match="Dot-based"):
        AddConfigVar(
            "T_config.something",
            doc="unittest",
            configparam=ConfigParam("invalid"),
            in_c_key=False,
        )


def test_mode_apply():

    assert configdefaults.filter_mode("DebugMode") == "DebugMode"

    with pytest.raises(ValueError, match="Expected one of"):
        configdefaults.filter_mode("not_a_mode")

    # test with theano.Mode instance
    import theano.compile.mode

    assert (
        configdefaults.filter_mode(theano.compile.mode.FAST_COMPILE)
        == theano.compile.mode.FAST_COMPILE
    )

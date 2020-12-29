"""Test config options."""
import configparser as stdlib_configparser
import io
import pickle

import pytest

from theano import configdefaults, configparser
from theano.configdefaults import short_platform
from theano.configparser import ConfigParam


def _create_test_config():
    return configparser.TheanoConfigParser(
        flags_dict={},
        theano_cfg=stdlib_configparser.ConfigParser(),
        theano_raw_cfg=stdlib_configparser.RawConfigParser(),
    )


def test_api_deprecation_warning():
    # accessing through configdefaults.config is the new best practice
    with pytest.warns(None):
        root = configdefaults.config
        assert isinstance(str(root), str)

    # accessing through configparser.config is discouraged
    root = configparser.config
    with pytest.warns(DeprecationWarning, match="instead"):
        root.add(
            "test_deprecationwarning",
            "A config var from a test case.",
            configparser.StrParam("test_default"),
        )
    with pytest.warns(DeprecationWarning, match="instead"):
        with root.change_flags(test_deprecationwarning="new_value"):
            pass


def test_api_redirect():
    root = _create_test_config()
    # one section level
    root.add(
        "test__section_redirect",
        "A config var from a test case.",
        configparser.StrParam("test_default"),
    )
    assert hasattr(root, "test__section_redirect")
    assert root.test__section_redirect == "test_default"
    assert hasattr(root, "test")
    assert isinstance(root.test, configparser._SectionRedirect)
    with pytest.warns(DeprecationWarning):
        assert root.test.section_redirect == "test_default"

    # two section levels
    root.add(
        "test__subsection__redirect",
        "A config var from a test case.",
        configparser.StrParam("test_default2"),
    )
    assert hasattr(root, "test__subsection__redirect")
    assert root.test__subsection__redirect == "test_default2"
    with pytest.warns(DeprecationWarning):
        assert root.test.subsection.redirect == "test_default2"


def test_invalid_default():
    # Ensure an invalid default value found in the Theano code only causes
    # a crash if it is not overridden by the user.

    root = _create_test_config()

    def validate(val):
        if val == "invalid":
            raise ValueError("Test-triggered")

    with pytest.raises(ValueError, match="Test-triggered"):
        # This should raise a ValueError because the default value is
        # invalid.
        root.add(
            "test__test_invalid_default_a",
            doc="unittest",
            configparam=ConfigParam("invalid", validate=validate),
            in_c_key=False,
        )

    root._flags_dict["test__test_invalid_default_b"] = "ok"
    # This should succeed since we defined a proper value, even
    # though the default was invalid.
    root.add(
        "test__test_invalid_default_b",
        doc="unittest",
        configparam=ConfigParam("invalid", validate=validate),
        in_c_key=False,
    )

    # Check that the flag has been removed
    assert "test__test_invalid_default_b" not in root._flags_dict


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
    root = _create_test_config()
    root.add(
        "test__config_hash",
        "A config var from a test case.",
        configparser.StrParam("test_default"),
    )

    h0 = root.get_config_hash()

    with root.change_flags(test__config_hash="new_value"):
        assert root.test__config_hash == "new_value"
        h1 = root.get_config_hash()

    h2 = root.get_config_hash()
    assert h1 != h0
    assert h2 == h0


def test_config_print():
    root = configdefaults.config
    result = str(root)
    assert isinstance(result, str)


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
    root = _create_test_config()
    root.add(
        "test__config_context",
        "A config var from a test case.",
        configparser.StrParam("test_default"),
    )
    assert hasattr(root, "test__config_context")
    assert root.test__config_context == "test_default"

    with root.change_flags(test__config_context="new_value"):
        assert root.test__config_context == "new_value"
        with root.change_flags({"test__config_context": "new_value2"}):
            assert root.test__config_context == "new_value2"
        assert root.test__config_context == "new_value"
    assert root.test__config_context == "test_default"


def test_invalid_configvar_access():
    root = configdefaults.config
    root_test = _create_test_config()

    # add a setting to the test instance
    root_test.add(
        "test__on_test_instance",
        "This config setting was added to the test instance.",
        configparser.IntParam(5),
    )
    assert hasattr(root_test, "test__on_test_instance")
    # While the property _actually_ exists on all instances,
    # accessing it through another instance raises an AttributeError.
    assert not hasattr(root, "test__on_test_instance")

    # But we can make sure that nothing crazy happens when we access it:
    with pytest.raises(configparser.ConfigAccessViolation, match="different instance"):
        print(root.test__on_test_instance)

    # And also that we can't add two configs of the same name to different instances:
    with pytest.raises(AttributeError, match="already registered"):
        root.add(
            "test__on_test_instance",
            "This config setting was already added to another instance.",
            configparser.IntParam(5),
        )


def test_no_more_dotting():
    root = configdefaults.config
    with pytest.raises(ValueError, match="Dot-based"):
        root.add(
            "test.something",
            doc="unittest",
            configparam=ConfigParam("invalid"),
            in_c_key=False,
        )


def test_mode_apply():

    assert configdefaults._filter_mode("DebugMode") == "DebugMode"

    with pytest.raises(ValueError, match="Expected one of"):
        configdefaults._filter_mode("not_a_mode")

    # test with theano.Mode instance
    import theano.compile.mode

    assert (
        configdefaults._filter_mode(theano.compile.mode.FAST_COMPILE)
        == theano.compile.mode.FAST_COMPILE
    )


def test_config_pickling():
    # check that the real thing can be pickled
    root = configdefaults.config
    buffer = io.BytesIO()
    pickle.dump(root, buffer)
    # and also unpickled...
    buffer.seek(0)
    restored = pickle.load(buffer)
    # ...without a change in the config values
    for name in root._config_var_dict.keys():
        v_original = getattr(root, name)
        v_restored = getattr(restored, name)
        assert (
            v_restored == v_original
        ), f"{name} did not survive pickling ({v_restored} != {v_original})"

    # and validate that the test would catch typical problems
    root = _create_test_config()
    root.add(
        "test__lambda_kills_pickling",
        "Lambda functions cause pickling problems.",
        configparser.IntParam(5, lambda i: i > 0),
    )
    with pytest.raises(AttributeError, match="Can't pickle local object"):
        pickle.dump(root, io.BytesIO())


class TestConfigHelperFunctions:
    @pytest.mark.parametrize(
        "release,platform,answer",
        [
            (
                "3.2.0-70-generic",
                "Linux-3.2.0-70-generic-x86_64-with-debian-wheezy-sid",
                "Linux-3.2--generic-x86_64-with-debian-wheezy-sid",
            ),
            (
                "3.2.0-70.1-generic",
                "Linux-3.2.0-70.1-generic-x86_64-with-debian-wheezy-sid",
                "Linux-3.2--generic-x86_64-with-debian-wheezy-sid",
            ),
            (
                "3.2.0-70.1.2-generic",
                "Linux-3.2.0-70.1.2-generic-x86_64-with-debian-wheezy-sid",
                "Linux-3.2--generic-x86_64-with-debian-wheezy-sid",
            ),
            (
                "2.6.35.14-106.fc14.x86_64",
                "Linux-2.6.35.14-106.fc14.x86_64-x86_64-with-fedora-14-Laughlin",
                "Linux-2.6-fc14.x86_64-x86_64-with-fedora-14-Laughlin",
            ),
        ],
    )
    def test_short_platform(self, release, platform, answer):
        o = short_platform(release, platform)
        assert o == answer, (o, answer)

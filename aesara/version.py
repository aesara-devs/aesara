try:
    from aesara._version import __version__ as version
except ImportError:
    raise RuntimeError(
        "Unable to find the version number that is generated when either building or "
        "installing from source. Please make sure that this Aesara has been properly "
        "installed, e.g. with\n\n  pip install -e .\n"
    )

deprecated_names = [
    "FALLBACK_VERSION",
    "full_version",
    "git_revision",
    "short_version",
    "release",
]


def __getattr__(name):
    # (Called when the module attribute is not found.)
    if name in deprecated_names:
        raise RuntimeError(
            f"{name} was deprecated when migrating away from versioneer. If you "
            f"need it, please search for or open an issue on GitHub entitled "
            f"'Restore deprecated versioneer variable {name}'.",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["version"]

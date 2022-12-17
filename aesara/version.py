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
            DeprecationWarning,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


version = "TODO"

import re
from pathlib import Path

__all__ = (
    "EditableProject",
    "__version__",
)

__version__ = "0.3"


# Check if a project name is valid, based on PEP 426:
# https://peps.python.org/pep-0426/#name
def is_valid(name):
    return (
        re.match(r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", name, re.IGNORECASE)
        is not None
    )


# Slightly modified version of the normalisation from PEP 503:
# https://peps.python.org/pep-0503/#normalized-names
# This version uses underscore, so that the result is more
# likely to be a valid import name
def normalize(name):
    return re.sub(r"[-_.]+", "_", name).lower()


class EditableException(Exception):
    pass


class EditableProject:
    def __init__(self, project_name, project_dir):
        if not is_valid(project_name):
            raise ValueError(f"Project name {project_name} is not valid")
        self.project_name = normalize(project_name)
        self.bootstrap = f"_editable_impl_{self.project_name}"
        self.project_dir = Path(project_dir)
        self.redirections = {}
        self.path_entries = []

    def make_absolute(self, path):
        return (self.project_dir / path).resolve()

    def map(self, name, target):
        if "." in name:
            raise EditableException(
                f"Cannot map {name} as it is not a top-level package"
            )
        abs_target = self.make_absolute(target)
        if abs_target.is_dir():
            abs_target = abs_target / "__init__.py"
        if abs_target.is_file():
            self.redirections[name] = str(abs_target)
        else:
            raise EditableException(f"{target} is not a valid Python package or module")

    def add_to_path(self, dirname):
        self.path_entries.append(self.make_absolute(dirname))

    def files(self):
        yield f"{self.project_name}.pth", self.pth_file()
        if self.redirections:
            yield f"{self.bootstrap}.py", self.bootstrap_file()

    def dependencies(self):
        deps = []
        if self.redirections:
            deps.append("editables")
        return deps

    def pth_file(self):
        lines = []
        if self.redirections:
            lines.append(f"import {self.bootstrap}")
        for entry in self.path_entries:
            lines.append(str(entry))
        return "\n".join(lines)

    def bootstrap_file(self):
        bootstrap = [
            "from editables.redirector import RedirectingFinder as F",
            "F.install()",
        ]
        for name, path in self.redirections.items():
            bootstrap.append(f"F.map_module({name!r}, {path!r})")
        return "\n".join(bootstrap)

import importlib.util
import sys


class RedirectingFinder:
    _redirections = {}

    @classmethod
    def map_module(cls, name, path):
        cls._redirections[name] = path

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if "." in fullname:
            return None
        if path is not None:
            return None
        try:
            redir = cls._redirections[fullname]
        except KeyError:
            return None
        spec = importlib.util.spec_from_file_location(fullname, redir)
        return spec

    @classmethod
    def install(cls):
        for f in sys.meta_path:
            if f == cls:
                break
        else:
            sys.meta_path.append(cls)

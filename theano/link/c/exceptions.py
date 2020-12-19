class MissingGXX(Exception):
    """
    This error is raised when we try to generate c code,
    but g++ is not available.

    """

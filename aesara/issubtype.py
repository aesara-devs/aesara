def issubtype(x, typ):
    if not isinstance(typ, tuple):
        typ = (typ,)

    for t in typ:
        if isinstance(x, type):
            if issubclass(x, t):
                return True
        else:
            if isinstance(x, typ):
                return True

    return False

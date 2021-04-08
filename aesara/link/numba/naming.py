_name_factory_stack = []

class NameFactory:
    def __init__(self):
        self._name_counts = {}
        
    def unique_name(self, name):
        count = self._name_counts.setdefault(name, 0)
        if count == 0:
            unique_name = name
        else:
            unique_name = self.unique_name(f"{name}_{count}")
        self._name_counts[name] += 1
        assert self._name_counts[unique_name] == 1
        return unique_name

    def __enter__(self):
        global _name_factory_stack
        _name_factory_stack.append(self)
    
    def __exit__(self, *args):
        global _name_factory_stack
        last = _name_factory_stack.pop()
        assert id(last) == id(self)


def unique_name(name):
    name = str(name)  # TODO
    if not _name_factory_stack:
        raise RuntimeError("No name factory on stack.")
    output = _name_factory_stack[-1].unique_name(name)
    #assert output.isidentifier(), output
    return output


def unique_name_for_apply(apply_node):
    if apply_node.name:
        return unique_name(apply_node.name)
    elif apply_node.owner is None:
        return unique_name(str(type(apply_node).__name__))
    else:
        return unique_name(str(type(apply_node.owner.op).__name__))
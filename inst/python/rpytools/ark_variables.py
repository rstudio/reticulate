from itertools import islice

MAX_DISPLAY = 100

def get_keys_and_children(inspector):
    keys, values = [], []
    keys_iterator = iter(inspector.get_children())
    # positron frontend only displays the first 100 items, then the length of the rest.
    try:
        for _ in range(MAX_DISPLAY):
            key = next(keys_iterator)
            keys.append(str(key))
            values.append(inspector.get_child(key))
    except StopIteration:
        return keys, values

    n_remaining = 0
    for n_remaining, _ in enumerate(keys_iterator, 1):
        pass
    if n_remaining > 0:
        keys.append("[[...]]")
        values.append(ChildrenOverflow(n_remaining))

    return keys, values


def get_child(inspector, index):
    key = next(islice(inspector.get_children(), index - 1, None))
    return inspector.get_child(key)

class ChildrenOverflow:
    def __init__(self, n_remaining):
        self.n_remaining = n_remaining

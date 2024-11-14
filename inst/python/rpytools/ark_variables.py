from itertools import islice

MAX_DISPLAY = 100


def get_keys_and_children(inspector):
    # positron frontend only displays the first 100 items, then the count of the rest.
    keys, values = [], []
    keys_iterator = iter(inspector.get_children())
    for key in islice(keys_iterator, MAX_DISPLAY):
        keys.append(str(key))
        values.append(inspector.get_child(key))

    if len(keys) == MAX_DISPLAY:
        # check if there are more children
        n_children = inspector.get_length()
        if n_children == 0:
            # no len() method, finish iteratoring over keys_iterator to get the true size
            for n_children, _ in enumerate(keys_iterator, MAX_DISPLAY + 1):
                pass
        n_remaining = n_children - MAX_DISPLAY
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

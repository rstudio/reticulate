from rpycall import call_r_function


def make_python_function(f, name=None):
    def python_function(*args, **kwargs):
        return call_r_function(f, *args, **kwargs)

    if name is not None:
        python_function.__name__ = name

    return python_function

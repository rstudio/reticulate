import rpycall


def make_python_function(f, name=None):
    def python_function(*args, **kwargs):
        # call the function
        value, error = rpycall.call_r_function(f, *args, **kwargs)

        if error:
            if isinstance(error, str) and error == "KeyboardInterrupt":
                # Only reachable if a C++ exception was caught in call_r_function()
                # otherwise error is always a bool()
                raise KeyboardInterrupt()

            if isinstance(error, Exception):
                raise error
            raise RuntimeError(error)
        return value

    if not name is None:
        python_function.__name__ = name

    return python_function

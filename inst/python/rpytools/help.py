import inspect


def normalize_func(func):
    # return None for builtins
    if inspect.isbuiltin(func):
        return None
    return func


def get_doc(func):
    doc = inspect.getdoc(func)
    if doc is None:
        func = normalize_func(func)
        if func is None:
            return None
        else:
            doc = inspect.getdoc(func)
    return doc


def get_property_doc(target, prop):
    for name, obj in inspect.getmembers(type(target), inspect.isdatadescriptor):
        if isinstance(obj, property) and name == prop:
            return inspect.getdoc(obj.fget)
    return None


def get_argspec(func):
    try:
        return inspect.getfullargspec(func)
    except TypeError:
        return None


def get_arguments(func):
    func = normalize_func(func)
    if func is None:
        return None
    argspec = get_argspec(func)
    if argspec is None:
        return None
    args = argspec.args
    if "self" in args:
        args.remove("self")
    return args


def get_r_representation(default):
    if callable(default) and hasattr(default, "__name__"):
        arg_value = default.__name__
    else:
        if default is None:
            arg_value = "NULL"
        elif type(default) == type(True):
            if default == True:
                arg_value = "TRUE"
            else:
                arg_value = "FALSE"
        elif isinstance(default, str):
            arg_value = '"%s"' % default
        elif isinstance(default, int):
            arg_value = "%rL" % default
        elif isinstance(default, float):
            arg_value = "%r" % default
        elif isinstance(default, (list, tuple, set)):
            constructor = "c" if isinstance(default, list) else "list"
            entries = ", ".join(get_r_representation(item) for item in default)
            arg_value = "%s(%s)" % (constructor, entries)
        elif isinstance(default, dict):
            dict_entries = ", ".join(
                [
                    ("%s = %s" % (k, get_r_representation(v)))
                    for k, v in default.items()
                ]
            )
            arg_value = "list(" + dict_entries + ")"
        else:
            arg_value = "%r" % default

    # if the value starts with "tf." then convert to $ usage
    if arg_value.startswith("tf."):
        arg_value = arg_value.replace(".", "$")

    return arg_value


def generate_signature_for_function(func):
    """Given a function, returns a string representing its args."""

    func = normalize_func(func)
    if func is None:
        return None

    args_list = []

    argspec = get_argspec(func)
    if argspec is None:
        return None

    first_arg_with_default = len(argspec.args or []) - len(
        argspec.defaults or []
    )
    for arg in argspec.args[:first_arg_with_default]:
        if arg == "self":
            # Python documentation typically skips `self` when printing method
            # signatures.
            continue
        args_list.append(arg)

    if argspec.defaults:
        for arg, default in zip(
            argspec.args[first_arg_with_default:], argspec.defaults
        ):
            arg_value = get_r_representation(default)
            args_list.append("%s = %s" % (arg, arg_value))
    if argspec.varargs:
        args_list.append("...")
    return "(" + ", ".join(args_list) + ")"

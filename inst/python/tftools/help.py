
import types
import inspect

def generate_signature_for_function(func):
    """Given a function, returns a string representing its args."""

    # convert class to __init__ method if we can
    if inspect.isclass(func):
      if (inspect.ismethod(func.__init__)):
        func = func.__init__
      else:
        return None

    args_list = []
    argspec = inspect.getargspec(func)
    first_arg_with_default = (
        len(argspec.args or []) - len(argspec.defaults or []))
    for arg in argspec.args[:first_arg_with_default]:
      if arg == "self":
        # Python documentation typically skips `self` when printing method
        # signatures.
        continue
      args_list.append(arg)

    # TODO(mrry): This is a workaround for documenting signature of
    # functions that have the @contextlib.contextmanager decorator.
    # We should do something better.
    if argspec.varargs == "args" and argspec.keywords == "kwds":
      original_func = func.__closure__[0].cell_contents
      return generate_signature_for_function(original_func)

    if argspec.defaults:
      for arg, default in zip(
          argspec.args[first_arg_with_default:], argspec.defaults):
        if callable(default):
          args_list.append("%s = %s" % (arg, default.__name__))
        else:
          if default is None:
            args_list.append("%s = NULL" % (arg))
          elif default == True:
            args_list.append("%s = TRUE" % (arg))
          elif default == False:
            args_list.append("%s = FALSE" % (arg))
          else:
            args_list.append("%s = %r" % (arg, default))
    if argspec.varargs:
      args_list.append("...")
    if argspec.keywords:
      args_list.append("...")
    return "(" + ", ".join(args_list) + ")"



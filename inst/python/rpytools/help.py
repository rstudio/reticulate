
import sys
import types
import inspect

def isstring(s):
    # if we use Python 3
    if (sys.version_info[0] >= 3):
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def normalize_func(func):
    # return None for builtins
    if (inspect.isbuiltin(func)):
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
    if (isinstance(obj, property) and name == prop):
      return inspect.getdoc(obj.fget)
  return None

def get_argspec(func):
  try:
    if sys.version_info[0] >= 3:
      return inspect.getfullargspec(func)
    else:
      return inspect.getargspec(func)
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
    if 'self' in args:
      args.remove('self')
    return args

def get_r_representation(default):
  if callable(default) and hasattr(default, '__name__'):
    arg_value = default.__name__
  else:
    if default is None:
      arg_value = "NULL"
    elif type(default) == type(True):
      if default == True:
        arg_value = "TRUE"
      else:
        arg_value = "FALSE"
    elif isstring(default):
      arg_value = "\"%s\"" % default
    elif isinstance(default, int):
      arg_value = "%rL" % default
    elif isinstance(default, float):
      arg_value = "%r" % default
    elif isinstance(default, list):
      arg_value = "c("
      for i, item in enumerate(default):
        if i is (len(default) - 1):
          arg_value += "%s)" % get_r_representation(item)
        else:
          arg_value += "%s, " % get_r_representation(item)
    elif isinstance(default, (tuple, set)):
      arg_value = "list("
      for i, item in enumerate(default):
        if i is (len(default) - 1):
          arg_value += "%s)" % get_r_representation(item)
        else:
          arg_value += "%s, " % get_r_representation(item)
    elif isinstance(default, dict):
      arg_value = "list("
      for i in range(len(default)):
        i_arg_value = "%s = %s" % \
          (default.keys()[i], get_r_representation(default.values()[i]))
        if i is (len(default) - 1):
          arg_value += "%s)" % i_arg_value
        else:
          arg_value += "%s, " % i_arg_value
    else:
      arg_value = "%r" % default
  
  # if the value starts with "tf." then convert to $ usage
  if (arg_value.startswith("tf.")):
    arg_value = arg_value.replace(".", "$")
      
  return(arg_value)

def generate_signature_for_function(func):
    """Given a function, returns a string representing its args."""

    func = normalize_func(func)
    if func is None:
      return None

    args_list = []
    
    argspec = get_argspec(func)
    if argspec is None:
      return None
  
    first_arg_with_default = (
        len(argspec.args or []) - len(argspec.defaults or []))
    for arg in argspec.args[:first_arg_with_default]:
      if arg == "self":
        # Python documentation typically skips `self` when printing method
        # signatures.
        continue
      args_list.append(arg)

    if argspec.varargs == "args" and hasattr(argspec, 'keywords') and argspec.keywords == "kwds":
      original_func = func.__closure__[0].cell_contents
      return generate_signature_for_function(original_func)

    if argspec.defaults:
      for arg, default in zip(
          argspec.args[first_arg_with_default:], argspec.defaults):
        arg_value = get_r_representation(default)
        args_list.append("%s = %s" % (arg, arg_value))
    if argspec.varargs:
      args_list.append("...")
    if hasattr(argspec, 'keywords') and argspec.keywords:
      args_list.append("...")
    return "(" + ", ".join(args_list) + ")"


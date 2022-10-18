
import sys
import threading

if sys.version_info.major < 3:
  import __builtin__ as builtins
else:
  import builtins


# the callback to be run
_callback = None

# a list of newly-imported packages
_imported_packages = []


# check for recursive imports
_recursion_depth = 0


# from builtins
__import__ = None


# from importlib
_find_and_load = None


# run hooks on packages to be imported
def _maybe_run_hooks():

  global _recursion_depth
  if _recursion_depth != 0:
    return False

  # check whether we can run our import hooks
  #
  # NOTE: Python code running on a separate thread might need to import
  # modules; if this occurs then we need to ensure that our R callback
  # is invoked only on the main thread
  is_main_thread = isinstance(threading.current_thread(), threading._MainThread)
  if not is_main_thread:
    return False

  global _imported_packages
  global _callback
  for package in _imported_packages:
    _callback(package)

  # remove imported packages from the list
  del _imported_packages[:]


# try to resolve a module name given the 'globals' and import level
def _resolve_module_name(name, globals=None, level=0):

  if level == 0:
    return name

  package = globals.get("__package__")
  if package is not None:
    return package

  spec = globals.get("__spec__")
  if spec is not None:
    return spec.parent


def _find_and_load_hook(name, import_):

  # check whether the module has already been imported
  already_imported = name in sys.modules

  # bump recursion depth
  global _recursion_depth
  _recursion_depth += 1

  # perform import
  try:
    global _find_and_load
    module = _find_and_load(name, import_)
  except:
    raise
  finally:
    _recursion_depth -= 1

  # if we haven't already imported this package, push
  # it onto the imported package list
  global _imported_packages
  if not already_imported:
    _imported_packages.append(name)

  # try and run hooks if possible
  _maybe_run_hooks()

  # return loaded module
  return module


# initialize the import hook using 'importlib'
def _initialize_importlib():

  import importlib._bootstrap

  # save original
  global _find_and_load
  _find_and_load = importlib._bootstrap._find_and_load

  # install our hook
  setattr(importlib._bootstrap, "_find_and_load", _find_and_load_hook)


# define our import hook
def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):

  # resolve module name
  resolved_module_name = _resolve_module_name(name, globals, level)

  # check whether the module has already been imported
  already_imported = resolved_module_name in sys.modules

  # bump the recursion level
  global _recursion_depth
  _recursion_depth += 1

  # perform the import
  try:
    global __import__
    module = __import__(name, globals=globals, locals=locals, fromlist=fromlist, level=level)
  except:
    raise
  finally:
    _recursion_depth -= 1

  # if we haven't already imported this package, push
  # it onto the imported package list
  global _imported_packages
  if not already_imported:
    _imported_packages.append(resolved_module_name)

  _maybe_run_hooks()

  return module

# the fallback initialization procedure
def _initialize_default():

  # save the original import implementation
  global __import__
  __import__ = builtins.__import__

  # apply our import hook
  builtins.__import__ = _import_hook



# the global entrypoint
def initialize(callback):

  global _callback
  _callback = callback

  try:
    import importlib._bootstrap
    if importlib._bootstrap is not None:
      return _initialize_importlib()
  except:
    pass

  return _initialize_default()

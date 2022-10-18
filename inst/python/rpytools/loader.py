
import sys
import threading

if sys.version_info.major < 3:
  import __builtin__ as builtins
else:
  import builtins


# The R callback to be run. Initialized in the 'initialize()' method.
_callback = None

# A list of Python packages which have been imported. The aforementioned
# callback will be run on the main thread after a module has been imported
# on the main thread.
_imported_packages = []

# A simple counter, tracking the recursion depth. This is used as we only
# attempt to run the R callback at the top level; that is, we don't want
# to run it while modules are being loaded recursively.
_recursion_depth = 0

# The builtin implementation of '__import__'; saved so that we can re-use it
# after initialization.
__import__ = builtins.__import__

# The implementation of '_find_and_load' captured from 'importlib._bootstrap'.
# Since we're trying to poke at Python internals, we try to wrap this code
# in try-catch and only use this if it appears safe to do so.
_find_and_load = None
try:
  import importlib._bootstrap
  _find_and_load = importlib._bootstrap._find_and_load
except:
  pass


# Run hooks on imported packages, if safe to do so.
def _maybe_run_hooks():

  # Don't run hooks while loading packages recursively.
  global _recursion_depth
  if _recursion_depth != 0:
    return False

  # Check whether we're on the main thread. Note that separate threads can
  # attempt to load Python modules, but the R callback we register can only
  # be safely run on the main thread.
  is_main_thread = isinstance(threading.current_thread(), threading._MainThread)
  if not is_main_thread:
    return False

  # Pre-flight checks passed; run the callbacks.
  global _imported_packages
  global _callback
  for package in _imported_packages:
    _callback(package)

  # Clear the import list.
  del _imported_packages[:]


# Resolve a module name on import. See Python code here for motivation.
# https://github.com/python/cpython/blob/c5140945c723ae6c4b7ee81ff720ac8ea4b52cfd/Lib/importlib/_bootstrap.py#L1246-L1270
def _resolve_module_name(name, globals=None, level=0):

  if level == 0:
    return name

  package = globals.get("__package__")
  if package is not None:
    return package

  spec = globals.get("__spec__")
  if spec is not None:
    return spec.parent
  
  return name

# Helper function for running an import hook with our extra scaffolding.
def _run_hook(name, hook):
  
  # Check whether this module has already been imported.
  already_imported = name in sys.modules

  # Bump the recursion depth.
  global _recursion_depth
  _recursion_depth += 1

  # Run the hook.
  try:
    module = hook()
  except:
    raise
  finally:
    _recursion_depth -= 1

  # Add this package to the import list, if this is the first
  # time importing that package.
  global _imported_packages
  if not already_imported:
    _imported_packages.append(name)

  # try and run hooks if possible
  _maybe_run_hooks()

  # return loaded module
  return module


# The hook installed to replace 'importlib._bootstrap._find_and_load'.
def _find_and_load_hook(name, import_):

  def _hook():
    global _find_and_load
    return _find_and_load(name, import_)
  
  return _run_hook(name, _hook)
  
# Initialize the '_find_and_load' replacement hook.
def _initialize_importlib():

  import importlib._bootstrap
  importlib._bootstrap._find_and_load = _find_and_load_hook
  

# The hook installed to replace '__import__'.
def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):

  # resolve module name
  resolved_module_name = _resolve_module_name(name, globals, level)
  
  def _hook():
    global __import__
    return __import__(name, globals=globals, locals=locals, fromlist=fromlist, level=level)
  
  return _run_hook(_hook)

# Initialize the '__import__' hook.
def _initialize_default():
  builtins.__import__ = _import_hook


# The main entrypoint for this module.
def initialize(callback):

  # Save the callback.
  global _callback
  _callback = callback

  # Check whether we can initialie with importlib.
  global _find_and_load
  if _find_and_load is not None:
    return _initialize_importlib()

  # Otherwise, fall back to default implementation.  
  return _initialize_default()

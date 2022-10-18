
import threading

# a list of newly-imported packages
_imported_packages = []

# check for recursive imports
_recursion_depth = 0

# adapted from:
# https://stackoverflow.com/questions/40623889/post-import-hooks-in-python-3
def initialize(callback):
  
  # NOTE: we try to import '__builtin__' first as 'builtins' is real
  # module provided by Python 2.x but it doesn't actually provide the
  # __import__ function definition!
  try:
    import __builtin__ as builtins  # python2.x
  except ImportError:
    import builtins  # python3.x

  import sys

  # save the original import implementation
  _import = builtins.__import__
  
  def _resolve_module_name(name, globals=None, level=0):
    
    if level == 0:
      return name
    
    package = globals.get("__package__")
    if package is not None:
      return package
    
    spec = globals.get("__spec__")
    if spec is not None:
      return spec.parent

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
      module = _import(
        name,
        globals=globals,
        locals=locals,
        fromlist=fromlist,
        level=level
      )
    except:
      raise
    finally:
      _recursion_depth -= 1
    
    # if we haven't already imported this package, push
    # it onto the imported package list
    global _imported_packages
    if not already_imported:
      _imported_packages.append(resolved_module_name)
    
    # check whether we can run our import hooks  
    #
    # NOTE: Python code running on a separate thread might need to import
    # modules; if this occurs then we need to ensure that our R callback
    # is invoked only on the main thread
    is_main_thread = isinstance(threading.current_thread(), threading._MainThread)
    run_hooks = _recursion_depth == 0 and is_main_thread
    
    # run our hooks if all safe 
    if run_hooks:
      [callback(package) for package in _imported_packages]
      del _imported_packages[:]
    
    return module

  # apply our import hook
  builtins.__import__ = _import_hook

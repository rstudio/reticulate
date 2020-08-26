
# adapted from:
# https://stackoverflow.com/questions/40623889/post-import-hooks-in-python-3
def initialize(callback):
  
  try:
    import builtins  # python3.x
  except ImportError:
    import __builtin__ as builtins  # python2.x

  import sys

  _import = builtins.__import__

  def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    
    already_imported = name in sys.modules

    module = _import(
      name,
      globals=globals,
      locals=locals,
      fromlist=fromlist,
      level=level
    )

    if not already_imported:
      callback(name)
    
    return module

  builtins.__import__ = _import_hook

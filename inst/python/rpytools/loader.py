
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

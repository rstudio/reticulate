
_callback = None

def on_import(module):
  global _callback
  _callback(module)
  
def initialize(callback):
  
  # save our default callback
  global _callback
  _callback = callback
  
  # define our import hook
  import importhook
  importhook.on_import(importhook.ANY_MODULE, on_import)
  

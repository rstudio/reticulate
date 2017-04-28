
import rpycall

def make_python_function(f, name = None):

  def python_function(*args, **kwargs):
    return rpycall.call_r_function(f, *args, **kwargs)

  if not name is None:
    python_function.__name__ = name
    
  return python_function

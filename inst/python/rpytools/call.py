
import rpycall

def make_python_function(f):

  def python_function(*args, **kwargs):
    return rpycall.call_r_function(f, *args, **kwargs)

  return python_function

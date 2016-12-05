
import tfcall

def make_python_function(f):

  def python_function(*args, **kwargs):
    return tfcall.call_r_function(f, *args, **kwargs)

  return python_function

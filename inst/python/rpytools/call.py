
import rpycall

# NOTE: these are also defined in python.cpp so must be changed in both places
kErrorKey = "F4B07A71E0ED40469929658827023424"
kInterruptError = "E04414EDEA17488B93FE2AE30F1F67AF";

def make_python_function(f, name = None):

  def python_function(*args, **kwargs):
    
    # call the function
    res = rpycall.call_r_function(f, *args, **kwargs)
    
    # check for an error
    if isinstance(res, dict) and kErrorKey in res:
      err = res[kErrorKey]
      if err == kInterruptError:
        raise KeyboardInterrupt()
      else:
        raise RuntimeError(res[kErrorKey])
      
    # otherwise return the result
    return res
    
  if not name is None:
    python_function.__name__ = name
    
  return python_function

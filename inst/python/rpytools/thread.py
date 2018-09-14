
import rpycall
import threading

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
  import Queue as queue
else:
  import queue as queue

def main_thread_func(f):

  def python_function(*args, **kwargs):

    if (isinstance(threading.current_thread(), threading._MainThread)):
      res = f(*args, **kwargs)
    else:
      result = queue.Queue()
      rpycall.call_python_function_on_main_thread(
        lambda: result.put(f(*args, **kwargs)),
        None
      )
      res = result.get()

    return res

  return python_function




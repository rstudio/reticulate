
import rpycall
import threading

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
  import Queue as queue
else:
  import queue as queue

class RGenerator(object):
  
  def __init__(self, r_function):
    self.r_function = r_function
  
  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if (isinstance(threading.current_thread(), threading._MainThread)):
      return self.r_function()
    else:
      result = queue.Queue()
      rpycall.call_python_function_on_main_thread(
        lambda: result.put(self.r_function()), 
        None
      )
      return result.get()


# Some test code

def iterate_on_thread(iter):
  def iteration_worker():
    for i in range(1,10):
      iter.next()
  thread = threading.Thread(target = iteration_worker)
  thread.start()
  while thread.isAlive():
    thread.join(0.1)





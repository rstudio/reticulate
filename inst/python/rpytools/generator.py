
import rpycall
import threading

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
  import Queue as queue
else:
  import queue as queue

class RGenerator(object):
  
  def __init__(self, r_function, completed):
    
    self.r_function = r_function
    self.completed = completed
  
  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    
    # call iterator
    if (isinstance(threading.current_thread(), threading._MainThread)):
      res = self.r_function()
    else:
      result = queue.Queue()
      rpycall.call_python_function_on_main_thread(
        lambda: result.put(self.r_function()), 
        None
      )
      res = result.get()
      
    # check for special 'completed' return value
    if (res == self.completed):
      raise StopIteration()
      
    # return result
    return res








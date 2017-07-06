
import rpycall
import threading

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
  import Queue as queue
else:
  import queue as queue

class RGenerator(object):
  
  def __init__(self, r_function, stop):
    
    self.r_function = r_function
    self.stop = stop
  
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
      
    # check for special 'stop' return value
    if (res == self.stop):
      raise StopIteration()
      
    # return result
    return res


# Some test code

def iterate_on_thread(iter):
  def iteration_worker():
    for i in iter:
      print i
  thread = threading.Thread(target = iteration_worker)
  thread.start()
  while thread.isAlive():
    thread.join(0.1)





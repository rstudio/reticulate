
import threading
import collections

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
  import Queue as queue
else:
  import queue as queue


def isScalar(x):
  return not isinstance(x, (list, tuple))

def isList(x):
  return isinstance(x, (list))

def asString(x):
  return str(x)

def makeDict():
  return {'a': 1.0, 'c': 3.0, 'b': 2.0}

def makeTuple():
  return (1.0, 2.0, 3.0)

def makeTupleWithOrderedDict():
  return (1.0, collections.OrderedDict({'b':777, 'a':22}))

def makeIterator(x):
  return iter(x)

def makeGenerator(n):
  i = 0
  while i < n:
    yield i
    i += 1
    
def iterateOnThread(iter):
  results = []
  def iteration_worker():
    for i in iter:
      results.append(i)
  thread = threading.Thread(target = iteration_worker)
  thread.start()
  while thread.is_alive():
    thread.join(0.1)
  return results

def invokeOnThread(f, *args, **kwargs):
  result = []
  def invoke_worker():
    result.append(f(*args, **kwargs))
  thread = threading.Thread(target = invoke_worker)
  thread.start()
  while thread.is_alive():
    thread.join(0.1)
  return result[0]


def reflect(x):
  return x

def callFunc(f, *args, **kwargs):
  return f(*args, **kwargs)

def testThrowError():
  throwError()

def throwError():
  raise ValueError('A very specific bad thing happened')


class PythonClass(object):
  
  FOO = 1
  BAR = 2
  
  @classmethod
  def class_method(cls):
    return cls.FOO
  
class PythonCallable(object):
  
  FOO = 1
  BAR = 2
  
  """ Call a callable
    Args:
      arg1: First argument.
  """
  def __call__(self, arg1):
    return arg1
  
def create_callable():
  return PythonCallable()

dict_with_callable = dict(callable = create_callable())




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

def makeIterator(x):
  return iter(x)

def makeGenerator(n):
  i = 0
  while i < n:
    yield i
    i += 1

def reflect(x):
  return x

def callFunc(f, *args, **kwargs):
  return f(*args, **kwargs)

def testThrowError():
  throwError()

def throwError():
  raise ValueError('A very specific bad thing happened')

def test_py_function_wrapper(arg1, arg2=2, arg3=[1,2], arg4=(1,2)):
  """Description for the function.
  Args:
    arg1: First argument.
    arg2: Second argument.
    arg3: Third argument.
    arg4: Fourth argument.
  Returns:
    `Some` object.
  Raises:
    ValueError: if something bad happens.
  """
  a = 1
  return a


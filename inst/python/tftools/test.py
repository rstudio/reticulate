

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

def reflect(x):
  return x

def callFunc(f, x):
  return f(x)

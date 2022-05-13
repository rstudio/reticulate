value = 42

def add(x, y):
  return x + y

def secret():
  return value

def _helper(): return 42
def api(): return _helper()


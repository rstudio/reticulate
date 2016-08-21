class Multiply:
  def __init__(self):
    self.a = 6
    self.b = 5

  def multiply(self):
    c = self.a*self.b
    print 'The result of', self.a, 'x', self.b, ':', c
    return c

  def multiply2(self, a, b):
    c = a*b
    print 'The result of', a, 'x', b, ':', c
    return c

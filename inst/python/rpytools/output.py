
import sys
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO

def start_stdout_capture():
  restore = sys.stdout
  sys.stdout = StringIO()
  return restore

def end_stdout_capture(restore):
  output = sys.stdout.getvalue()
  sys.stdout.close()
  sys.stdout = restore
  return output

def start_stderr_capture():
  restore = sys.stderr
  sys.stderr = StringIO()
  return restore

def end_stderr_capture(restore):
  output = sys.stderr.getvalue()
  sys.stderr.close()
  sys.stderr = restore
  return output


class OutputRemap(object):
  
  def __init__(self, target, handler):
    self.target = target
    self.handler = handler
  
  def write(self, message):
    self.handler(message)
    
  def __getattr__(self, attr): 
    if (self.target): 
      return getattr(self.target, attr)
    else:
      return 0

  def close(self):
    return None
  
  def flush(self):
    return None


def remap_output_streams(r_stdout, r_stderr):
  if (sys.stdout is None):
    sys.stdout = OutputRemap(sys.stdout, r_stdout)
  if (sys.stderr is None):
    sys.stderr = OutputRemap(sys.stderr, r_stderr)






  

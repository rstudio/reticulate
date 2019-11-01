
import sys
import io

_capture_stdout = io.StringIO()
_capture_stderr = io.StringIO()
_stdout  = None
_stderr  = None

def start_capture(capture_stdout, capture_stderr):
  
  global _stdout
  global _stderr
  
  if capture_stdout:
    _stdout = sys.stdout
    sys.stdout = _capture_stdout
    
  if capture_stderr:
    _stderr = sys.stderr
    sys.stderr = _capture_stderr

def end_capture():
  
  global _stdout
  global _stderr
  
  if _stdout is not None:
    _capture_stdout.seek(0)
    _capture_stdout.truncate()
    sys.stdout = _stdout
    _stdout = None
    
  if _stderr is not None:
    _capture_stderr.seek(0)
    _capture_stderr.truncate()
    sys.stderr = _stderr
    _stderr = None
  
def collect_output():
  
  global _stdout
  global _stderr
  
  # collect outputs into array
  outputs = []
  if _stdout is not None:
    stdout = _capture_stdout.getvalue()
    if stdout:
      outputs.append(stdout)
      
  if _stderr is not None:
    stderr = _capture_stderr.getvalue()
    if stderr:
      outputs.append(stderr)
    
  # ensure trailing newline
  outputs.append('')
  
  # join outputs
  return '\n'.join(outputs)
  
class OutputRemap(object):
  
  def __init__(self, target, handler, tty = True):
    self.target = target
    self.handler = handler
    self.tty = tty
  
  def write(self, message):
    return self.handler(message)
    
  def isatty(self):
    return self.tty
    
  def __getattr__(self, attr): 
    if (self.target): 
      return getattr(self.target, attr)
    else:
      return 0

  def close(self):
    return None
  
  def flush(self):
    return None


def remap_output_streams(r_stdout, r_stderr, tty, force):
  if (force or sys.stdout is None):
    sys.stdout = OutputRemap(sys.stdout, r_stdout, tty)
  if (force or sys.stderr is None):
    sys.stderr = OutputRemap(sys.stderr, r_stderr, tty)






  

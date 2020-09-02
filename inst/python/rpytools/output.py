
# NOTE: indirection required here for Python 2.7 support
# TypeError: unicode argument expected, got 'str'
import sys
if sys.version_info < (3, 0):
  from io import BytesIO as StringIO
else:
  from io import StringIO

_capture_stdout = StringIO()
_capture_stderr = StringIO()
_stdout = None
_stderr = None

def _override_logger_streams(capture_stdout, old_stdout, new_stdout,
                             capture_stderr, old_stderr, new_stderr):
  import logging
  
  # capture root handlers
  root = getattr(logging, 'root', None)
  if root is not None:
    handlers = getattr(root, 'handlers', [])
    for handler in handlers:
      
      stream = getattr(handler, 'stream', None)
      if stream is None:
        continue
      
      if capture_stdout and stream is old_stdout:
        handler.stream = new_stdout
        
      if capture_stderr and stream is old_stderr:
        handler.stream = new_stderr
  
  # capture loggers registered with the default manager
  loggers = getattr(logging.Logger.manager, 'loggerDict', {})
  for logger in loggers.values():
    handlers = getattr(logger, 'handlers', [])
    for handler in handlers:
      
      stream = getattr(handler, 'stream', None)
      if stream is None:
        continue
      
      if capture_stdout and handler.stream is old_stdout:
        handler.stream = new_stdout
        
      if capture_stderr and handler.stream is old_stderr:
        handler.stream = new_stderr
          
def start_capture(capture_stdout, capture_stderr):
  
  global _stdout
  global _stderr
  
  if capture_stdout:
    _stdout = sys.stdout
    sys.stdout = _capture_stdout
    
  if capture_stderr:
    _stderr = sys.stderr
    sys.stderr = _capture_stderr
  
  try:
    _override_logger_streams(
      capture_stdout, sys.__stdout__, _capture_stdout,
      capture_stderr, sys.__stderr__, _capture_stderr
    )
  except:
    pass

def end_capture(capture_stdout, capture_stderr):
  
  global _stdout
  global _stderr
  
  if capture_stdout:
    _capture_stdout.seek(0)
    _capture_stdout.truncate()
    sys.stdout = _stdout
    _stdout = None
    
  if capture_stderr:
    _capture_stderr.seek(0)
    _capture_stderr.truncate()
    sys.stderr = _stderr
    _stderr = None
    
  try:
    _override_logger_streams(
      capture_stdout, _capture_stdout, sys.__stdout__,
      capture_stderr, _capture_stderr, sys.__stderr__
    )
  except:
    pass
  
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






  

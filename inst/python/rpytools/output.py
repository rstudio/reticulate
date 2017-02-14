
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

  

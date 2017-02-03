
import sys
from cStringIO import StringIO

def start_stdout_capture():
  restore = sys.stdout
  sys.stdout = StringIO()
  return restore

def end_stdout_capture(restore):
  output = sys.stdout.getvalue()
  sys.stdout.close()
  sys.stdout = restore
  return output


import signal
import sys

_handler = None

def signal_handler(sig, frame):
  _handler(sig)
  raise KeyboardInterrupt
  
def initialize(handler):
  
  # initialize R handler
  global _handler
  _handler = handler;
  
  # register interrupt handlers
  signal.signal(signal.SIGINT, signal_handler)

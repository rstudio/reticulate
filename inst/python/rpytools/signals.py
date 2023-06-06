
# We register a custom Python signal handler here for the following reasons.
#
#  1. We are running R and Python together within the same process.
#     Interrupts should be handled by whichever process appears to
#     be at the foreground.
#
#  2. To facilitate this, we need to register a SIGINT handler at
#     the C level. We cannot rely on the Python signal handler
#     directly, since the Python runtime won't get a chance to
#     to process the interrupt if R is at the foreground.
#
#  3. However, newer versions of Python get upset if the default
#     signal handler is tripped by a custom signal handler.
#
#  4. However, we cannot easily tell whether R or Python is at the foreground.
#     To resolve this, we allow _both_ Python and R to "wake up" and prepare
#     to handle the interrupt, but allow Python to bail out in the signal
#     handler if it sees that R handled the interrupt first.
#
from signal import signal, SIGINT

# The "are interrupts pending?" callback. The intention here is that, when
# an interrupt is signaled, both R and Python will then "race" to see who
# can handle the interrupt first. To that end, we want Python to also respect
# the interrupt flag set by R. If the Python interrupt handler gets a chance to
# run _after_ the interrupt has been handled by R, then we want to just ignore
# that interrupt. However, if Python sees it _before_ R, then Python should
# handle the interrupt, and tell R the interrupt was handled (so R doesn't
# try to also handle it).
#
# It's not yet clear to me whether this is the correct strategy for cases
# where R and Python are calling back to each other recursively.
_callback = None

def _signal_handler(sig, frame):
  
  # Ask R whether interrupts are pending.
  global _callback
  pending = _callback(False)
  
  # If interrupts are pending, now's our chance to handle it.
  # Now that Python is getting a chance to handle interrupts,
  # we can tell R to unset the interrupts pending flag.
  if pending:
    _callback(True)
    raise KeyboardInterrupt
  
def initialize(callback):

  # Initialize our callback.  
  global _callback
  _callback = callback
  
  # Set our signal handler.
  signal(SIGINT, _signal_handler)
  

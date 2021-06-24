
# I know what you're thinking. Why does this exist? It looks like we're
# just installing a signal handler that does the same thing as the "default"
# Python SIGINT signal handler, so why not just rely on the default behavior?
#
# To answer your question, because:
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
# The solution, then, is to provide a custom signal handler that behaves
# the same as the default signal handler, since the aforementioned code
# will then no longer complain when that handler is found and invoked.
#
# See https://github.com/python/cpython/blob/bd4ab8e73906a4f12d5353f567228b7c7497baf7/Modules/signalmodule.c#L1715-L1737
# for more details.
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
  
  # Now that Python is getting a chance to handle interrupts,
  # we can tell R to unset the interrupts pending flag.
  _callback(True)
  
  # If interrupts are pending, now's our chance to handle it.
  if pending:
    raise KeyboardInterrupt
  
def initialize(callback):

  # Initialize our callback.  
  global _callback
  _callback = callback
  
  # Set our signal handler.
  signal(SIGINT, _signal_handler)
  

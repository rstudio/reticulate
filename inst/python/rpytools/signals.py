
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
import signal
import sys

def _signal_handler(sig, frame):
  raise KeyboardInterrupt
  
def initialize():
  signal.signal(signal.SIGINT, _signal_handler)

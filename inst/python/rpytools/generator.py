import rpycall
import threading

import sys

is_py2 = sys.version[0] == "2"
if is_py2:
    import Queue as queue
else:
    import queue as queue

_completed_sentinel = object()


class RGenerator(object):
    def __init__(self, r_function, prefetch=0):
        self.r_function = r_function
        self.prefetch = prefetch
        self.values_queue = queue.Queue()
        self.completed = False
        self._pending_tend_queue = False
        if prefetch:
            self._tend_queue()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self._tend_queue(1)
        val = self._get_or_raise()
        if self.prefetch:
            self._tend_queue()
        return val

    def _tend_queue(self, min_fetch=0):
        if self.completed:
            return

        ## Prefetch and enqueue generator generated values.

        # If we're not on the main thread, make sure there is a pending call to
        # this function from the main thread and return.
        if threading.current_thread() is not threading.main_thread():
            if not self._pending_tend_queue:
                self._pending_tend_queue = True
                rpycall.call_python_function_on_main_thread(self._tend_queue, min_fetch)
            return

        # We're on the main thread, call the generator and put values on the queue.
        self._pending_tend_queue = False

        fetch = max(min_fetch, self.prefetch - self.values_queue.qsize())
        for _ in range(fetch):
            try:
                val = self.r_function()
            except StopIteration:
                self.values_queue.put(_completed_sentinel)
                self.completed = True
                return

            self.values_queue.put(val)

    def _get_or_raise(self):
        try:
            # only wait/block-thread/yield-to-another-thread if the R generator
            # is not exhausted.
            val = self.values_queue.get(block=~self.completed)
            if val is _completed_sentinel:
                raise StopIteration()
            return val
        except queue.Empty:
            # only get here if self.completed = True and the queue is empty,
            # meaning we've already pulled off _completed_sentinel and
            # raised StopIteration
            raise StopIteration()

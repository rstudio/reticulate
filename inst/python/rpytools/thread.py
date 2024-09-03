import sys
import queue as queue
from rpycall import call_r_function, schedule_python_function_on_main_thread


def safe_call_r_function(f, args, kwargs):
    try:
        return call_r_function(f, *args, **kwargs), None
    except Exception as e:  # TODO: should we catch BaseException too? KeyboardInterrupt?
        return None, e


def safe_call_r_function_on_main_thread(f, *args, **kwargs):
    result = queue.Queue()
    schedule_python_function_on_main_thread(
        lambda: result.put(safe_call_r_function(f, args, kwargs)),
        None,
    )
    return result.get()

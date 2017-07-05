
import rpycall

def register_task(func, data = None):
  rpycall.register_event_loop_task(func, data)

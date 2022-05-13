import IPython
from traitlets.config import Config


_c = Config()

_c.InteractiveShell.confirm_exit = False
_c.TerminalIPythonApp.display_banner = False

# c.InteractiveShell.colors = 'Neutral'
# 'Neutral', 'NoColor', 'LightBG', 'Linux'

# Only need to register callbacks on first init
# There is probably a better way to not register the same callback multiple times
# doing a straight comparison like `fn in callbacks_list` fails because
# ipython decorates the registered callbacks
_c.InteractiveShellApp.exec_lines = [
    """
def _reticulate_init():
  import sys
  ev = get_ipython().events
  if not any(fn.__name__ == 'flush' for fn in ev.callbacks['post_run_cell']):
      ev.register("post_run_cell", sys.stdout.flush)
      ev.register("post_run_cell", sys.stderr.flush)

_reticulate_init()
del _reticulate_init
"""
]


def start_ipython():
    IPython.start_ipython(config=_c)

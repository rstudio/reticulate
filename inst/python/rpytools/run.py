import sys
import os


def run_file(path):
    with open(path, "r") as file:
        file_content = file.read()

    d = sys.modules["__main__"].__dict__

    exec(file_content, d, d)


class RunMainScriptContext:
    def __init__(self, path, args):
        self.path = path
        self.args = tuple(args)

    def __enter__(self):
        sys.path.insert(0, os.path.dirname(self.path))

        self._orig_sys_argv = sys.argv
        sys.argv = [self.path] + list(self.args)

    def __exit__(self, *_):
        # try restore sys.path
        try:
            sys.path.remove(os.path.dirname(self.path))
        except ValueError:
            pass
        # restore sys.argv if it's been unmodified
        # otherwise, leave it as-is.
        set_argv = [self.path] + list(self.args)
        if sys.argv == set_argv:
            sys.argv = self._orig_sys_argv


def _run_file_on_thread(path, args=None):

    import _thread

    _thread.start_new_thread(run_file, (path, args))


def _launch_lsp_server_on_thread(path, args):
    # for now, leave sys.argv and sys.path permanently modified.
    # Later, revisit if it's desirable/safe to restore after the initial
    # lsp event loop startup.
    RunMainScriptContext(path, args).__enter__()
    _run_file_on_thread(path)

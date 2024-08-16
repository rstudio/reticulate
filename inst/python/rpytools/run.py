import sys
import os


def run_file(path):
    with open(path, "r") as file:
        file_content = file.read()

    # ipykernel patches the loader, so `import __main__` does not
    # produce the "real" __main__
    d = sys.modules["__main__"].__dict__

    exec(file_content, d, d)


class RunMainScriptContext:
    def __init__(self, path, argv=None):
        self.path = path
        self.argv = tuple(argv) if argv is not None else None

    def __enter__(self):
        sys.path.insert(0, os.path.dirname(self.path))

        if self.argv is not None:
            self._orig_sys_argv = sys.argv
            sys.argv = [self.path] + list(self.argv)

    def __exit__(self, *_):
        # try restore sys.path
        try:
            sys.path.remove(os.path.dirname(self.path))
        except ValueError:
            pass
        # try restore sys.argv if we patched it
        if self.argv is not None:
            # restore sys.argv if it's unmodified from what we set it to.
            # otherwise, leave it as-is.
            patched_argv = [self.path] + list(self.args)
            if sys.argv == patched_argv:
                sys.argv = self._orig_sys_argv


def _run_file_on_thread(path, args=None):

    import _thread

    _thread.start_new_thread(run_file, (path,))


def _launch_lsp_server_on_thread(path, args):
    # for now, leave sys.argv and sys.path permanently modified.
    # Later, revisit if it's desirable/safe to restore after the initial
    # lsp event loop startup.
    RunMainScriptContext(path, args).__enter__()
    _run_file_on_thread(path)


def run_file_on_thread(path, args=None):
    RunMainScriptContext(path, args).__enter__()
    _run_file_on_thread(path)

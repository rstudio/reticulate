# NOTE: indirection required here for Python 2.7 support
# TypeError: unicode argument expected, got 'str'
import sys

if sys.version_info < (3, 0):
    from io import BytesIO as StringIO
else:
    from io import StringIO


def _setStream(handler, stream):
    setStream = getattr(handler, "setStream", None)
    if setStream is not None:
        return setStream(stream)
    old_stream = handler.stream
    handler.stream = stream
    return old_stream


def _override_logger_streams(
    capture_stdout,
    old_stdout,
    new_stdout,
    capture_stderr,
    old_stderr,
    new_stderr,
):
    import logging

    # capture root handlers
    root = getattr(logging, "root", None)
    if root is not None:
        handlers = getattr(root, "handlers", [])
        for handler in handlers:

            stream = getattr(handler, "stream", None)
            if stream is None:
                continue
            if capture_stdout and stream is old_stdout:
                _setStream(handler, new_stdout)
            elif capture_stderr and stream is old_stderr:
                _setStream(handler, new_stderr)
    # capture loggers registered with the default manager
    loggers = getattr(logging.Logger.manager, "loggerDict", {})
    for logger in loggers.values():
        handlers = getattr(logger, "handlers", [])
        for handler in handlers:
            stream = getattr(handler, "stream", None)
            if stream is None:
                continue
            if capture_stdout and stream is old_stdout:
                _setStream(handler, new_stdout)
            elif capture_stderr and stream is old_stderr:
                _setStream(handler, new_stderr)


class OutputCaptureContext:
    def __init__(self, capture_stdout, capture_stderr):
        self._capture_stdout = bool(capture_stdout)
        self._capture_stderr = bool(capture_stderr)
        self._capturing_stream = StringIO()

    def __enter__(self):
        if self._capture_stdout:
            self._prev_stdout = sys.stdout
            sys.stdout = self._capturing_stream

        if self._capture_stderr:
            self._prev_stderr = sys.stderr
            sys.stderr = self._capturing_stream

        _override_logger_streams(
            capture_stdout=self._capture_stdout,
            new_stdout=sys.stdout if self._capture_stdout else None,
            old_stdout=self._prev_stdout if self._capture_stdout else None,
            capture_stderr=self._capture_stderr,
            new_stderr=sys.stderr if self._capture_stderr else None,
            old_stderr=self._prev_stderr if self._capture_stderr else None,
        )

    def __exit__(self, *args):
        if self._capture_stdout:
            sys.stdout = self._prev_stdout
        if self._capture_stderr:
            sys.stderr = self._prev_stderr

        _override_logger_streams(
            capture_stdout=self._capture_stdout,
            new_stdout=sys.stdout,
            old_stdout=self._prev_stdout if self._capture_stdout else None,
            capture_stderr=self._capture_stderr,
            new_stderr=sys.stderr,
            old_stderr=self._prev_stderr if self._capture_stderr else None,
        )

    def collect_output(self, clear=True):
        output = self._capturing_stream.getvalue()
        if clear:
            self._capturing_stream.truncate(0)
            self._capturing_stream.seek(0)
        return output


class OutputRemap(object):

    def __init__(self, target, handler, tty=True):
        self.target = target
        self.handler = handler
        self.tty = tty

    def write(self, message):
        return self.handler(message)

    def isatty(self):
        return self.tty

    def __getattr__(self, attr):
        if self.target:
            return getattr(self.target, attr)
        else:
            return 0

    def close(self):
        return None

    def flush(self):
        return None


def _remap_output_streams(r_stdout, r_stderr, tty):
    sys.stdout = OutputRemap(sys.stdout, r_stdout, tty)
    sys.stderr = OutputRemap(sys.stderr, r_stderr, tty)


class RemapOutputStreams:
    def __init__(self, r_stdout, r_stderr, tty):
        self.r_stdout = r_stdout
        self.r_stderr = r_stderr
        self.tty = tty
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        # It's possible that __enter__ does not execute before __exit__ in some
        # special cases. We also store _stdout and _stderr when creating the context.
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        _remap_output_streams(self.r_stdout, self.r_stderr, self.tty)

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

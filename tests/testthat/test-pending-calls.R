test_that("background R callbacks recover from a full Python pending-call queue", {
  skip_if_no_python()

  withr::local_envvar(RETICULATE_PYTHON = py_exe())
  result <- callr::r(function() {
    library(reticulate)
    main <- import_main()
    main$callback <- function() 42L
    py_run_string("
import ctypes
import sys
import threading

pending_call_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
add_pending_call = ctypes.pythonapi.Py_AddPendingCall
add_pending_call.argtypes = [pending_call_type, ctypes.c_void_p]
add_pending_call.restype = ctypes.c_int

main_thread = threading.get_ident()
drained = []

@pending_call_type
def pending_call(data: object) -> int:
    drained.append(threading.get_ident())
    return 0

start = threading.Event()
ready = threading.Event()
queue_full = threading.Event()
completed = threading.Event()
accepted = []
results = []

def worker() -> None:
    # Initialize the callback bridge before filling the queue: importing its
    # Python module can release the GIL and let the main thread drain calls.
    results.append(callback())
    ready.set()
    start.wait()
    while add_pending_call(pending_call, None) == 0:
        accepted.append(None)
    queue_full.set()
    try:
        results.append(callback())
    except BaseException as error:
        results.append(error)
    finally:
        completed.set()

# Keep the worker's GIL from being released between filling the queue and
# calling R. Only the scheduler's retry can let the main thread drain it.
previous_interval = sys.getswitchinterval()
thread = threading.Thread(target=worker, daemon=True)
thread.start()
try:
    while not ready.wait(0.01):
        pass
    sys.setswitchinterval(60)
    start.set()
    queue_full.wait()
    while not completed.wait(0.01):
        pass
    thread.join()
finally:
    sys.setswitchinterval(previous_interval)

assert accepted
assert drained == [main_thread] * len(accepted)
assert results == [42, 42]
")
    TRUE
  }, timeout = 15)

  expect_true(result)
})

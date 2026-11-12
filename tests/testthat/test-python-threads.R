

test_that("py_allow_threads() can enable/disable background threads", {

  file <- tempfile()
  on.exit(unlink(file), add = TRUE)

  write_to_file_from_thread <- py_run_string("
def write_to_file_from_thread(path, lines):
    from time import sleep, localtime, strftime

    def write_to_file(path, lines):
        sleep(.1) # don't try to run until we've had a chance to return to the R main thread
        with open(path, 'w') as f:
            for line in list(lines):
                f.write(line + '\\n')

    from _thread import start_new_thread
    start_new_thread(write_to_file, (path, lines))
", local = TRUE)$write_to_file_from_thread

  reticulate:::py_allow_threads_impl(FALSE)
  write_to_file_from_thread(file, letters)
  Sys.sleep(.5)
  # confirm background thread did not run while R was sleeping
  expect_false(file.exists(file))
  # explicitly enter python and release the gil
  import("time")$sleep(.3)
  # confirm the background thread ran on py_sleep()
  expect_identical(readLines(file), letters)

  unlink(file)

  reticulate:::py_allow_threads_impl(TRUE)
  write_to_file_from_thread(file, letters)
  Sys.sleep(.3)
  # confirm that the background thread ran while R was sleeping.
  expect_identical(readLines(file), letters)

})



test_that("Python calls into R from a background thread are evaluated", {

  x <- 0L
  r_func <- function() x <<- x+1
  py_file <- withr::local_tempfile(lines = "r_func()", fileext = ".py")

  reticulate:::py_run_file_on_thread(
    py_file,
    init_globals = list(r_func = r_func)
  )

  # Simulate the main R thread doing non-Python work (e.g., sleeping)
  for(i in 1:10) {
    Sys.sleep(.01 * i)
    if (x != 0L) break
  }

  expect_equal(x, 1L)
})


test_that("Errors from background threads calling into main thread are handled", {

  signal_r_error <- function() stop("foo-bar-baz")

  val <- NULL
  set_val <- function(v) val <<- v

  py_file <- withr::local_tempfile(lines = "
try: signal_r_error()
except Exception as e: set_val(e.args[0])
", fileext = ".py")

  reticulate:::py_run_file_on_thread(py_file, init_globals = list(
    signal_r_error = signal_r_error,
    set_val = set_val
  ))

  # Simulate the main R thread doing non-Python work (e.g., sleeping)
  for(i in 1:10) {
    Sys.sleep(.01 * i)
    if(!is.null(val)) break
  }

  expect(!is.null(val), "`thread_err_msg` never created")
  expect_equal(val, "foo-bar-baz")

})


for (kind in c("value", "RuntimeError", "KeyboardInterrupt", "SystemExit")) {
  test_that(paste("Background R callbacks deliver", kind, "to their caller"), {
    skip_on_cran()
    result <- callr::r(function(python, kind) {
      Sys.setenv(RETICULATE_PYTHON = python)
      library(reticulate)
      main <- import_main()

      main$callback <- switch(kind,
        value = function() 42L,
        RuntimeError = function() stop("callback failed"),
        KeyboardInterrupt = function() rlang::interrupt(),
        SystemExit = function() stop(import_builtins()$SystemExit("cancelled"))
      )
      main$success <- function() 42L

      # Keep executing Python on the main thread so pending R calls can run.
      # The subprocess timeout also contains a regression that strands a worker.
      py_run_string("
import threading

outcome = None
done = threading.Event()

def worker() -> None:
    global outcome
    try:
        outcome = ['value', callback()]
    except BaseException as exc:
        outcome = [type(exc).__name__, str(exc)]
    finally:
        done.set()

thread = threading.Thread(target=worker, daemon=True)
thread.start()
while not done.wait(0.01):
    pass
thread.join()

# A cancelled callback must not prevent the next callback from completing.
callback = success
done.clear()
first_outcome = outcome
thread = threading.Thread(target=worker, daemon=True)
thread.start()
while not done.wait(0.01):
    pass
thread.join()

assert outcome == ['value', 42]
")
      main$first_outcome
    }, args = list(python = py_exe(), kind = kind), timeout = 15)

    expect_identical(result[[1]], kind)
    expected <- switch(kind,
      value = 42L,
      RuntimeError = "callback failed",
      KeyboardInterrupt = "",
      SystemExit = "cancelled"
    )
    expect_identical(result[[2]], expected)
  })
}

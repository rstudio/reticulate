

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
  py$r_func <- function() x <<- x+1
  on.exit(py_del_attr(py, "r_func"), add = TRUE)
  py_file <- withr::local_tempfile(lines = "r_func()", fileext = ".py")

  reticulate:::py_run_file_on_thread(py_file)

  # Simulate the main R thread doing non-Python work (e.g., sleeping)
  for(i in 1:10) {
    Sys.sleep(.01 * i)
    if (x != 0L) break
  }

  expect_equal(x, 1L)
})


test_that("Errors from background threads calling into main thread are handled", {

  py$signal_r_error <- function() stop("foo-bar-baz")
  on.exit(py_del_attr(py, "signal_r_error"), add = TRUE)

  # {testthat} messes with the globalenv() somehow during `test_that()`,
  # the test fails if we try to communicate by assigning in the globalenv via
  #   r.val = 'foo'
  val <- NULL
  py$set_val <-  function(v) val <<- v
  on.exit(py_del_attr(py, "set_val"), add = TRUE)

  py_file <- withr::local_tempfile(lines = "
try: signal_r_error()
except Exception as e: set_val(e.args[0])
", fileext = ".py")

  reticulate:::py_run_file_on_thread(py_file)

  # Simulate the main R thread doing non-Python work (e.g., sleeping)
  for(i in 1:10) {
    Sys.sleep(.01 * i)
    if(!is.null(val)) break
  }

  expect(!is.null(val), "`thread_err_msg` never created")
  expect_equal(val, "foo-bar-baz")

})

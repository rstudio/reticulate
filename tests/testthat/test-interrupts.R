
test_that("Running Python scripts can be interrupted", {

  skip_on_cran()

  p <- callr::r_bg(args = list(python = py_exe()), function(python) {
    Sys.setenv(RETICULATE_PYTHON = python)
    library(reticulate)

    tryCatch({
      py_run_string("
import time
print('Sleeping', flush=True)
time.sleep(5)
print('Sleep completed', flush=True)
")
      FALSE
    }, interrupt = function(e) TRUE)
  })
  on.exit(p$kill(), add = TRUE)

  # Wait for Python's checkpoint, independently of process startup time.
  while (!length(checkpoint <- p$read_output_lines(1)) && p$is_alive())
    callr::poll(list(p$get_output_connection()), -1)
  expect_identical(checkpoint, "Sleeping")

  p$interrupt()
  p$wait()

  expect_identical(p$get_exit_status(), 0L)
  expect_true(p$get_result())
  expect_identical(p$read_all_output(), "")
})


test_that("interrupts work when Python is running", {

  skip_on_cran()

  p <- callr::r_bg(args = list(python = py_exe()), function(python) {
    library(reticulate)
    use_python(python)
    get_frames <- function() {
      r_stack <- sys.calls()
      py_stack <- py_capture_output(py_run_string("from traceback import print_stack; print_stack()"))
      list(r = r_stack, py = py_stack)
    }
    frames_before <- get_frames()

    py_run_string("print('Initialized Python')")
    tryCatch({
      py_run_string(glue::trim("
        print('Starting', flush=True)
        i = 0
        while True:
          i += 1
        "))
    }, interrupt = function(e) {
      cat("Caught interrupt; ")
    })

    # confirm that the python stack was unwound correctly
    frames_after <- get_frames()
    stopifnot(identical(frames_before, frames_after))

    cat("Finished!")
  })


  p$poll_io(5000)
  expect_identical(p$read_output_lines(1), "Initialized Python")

  p$poll_io(500)
  expect_identical(p$read_output_lines(1), "Starting")

  p$poll_io(500)
  expect_identical(p$read_output(), "")

  p$interrupt()
  p$wait()

  expect_identical(p$get_exit_status(), 0L)
  expect_identical(p$read_all_output(), "Caught interrupt; Finished!")

})


test_that("interrupts can be caught by Python", {
  skip_on_cran()

  p <- callr::r_bg(args = list(python = py_exe()), function(python) {

    Sys.setenv(RETICULATE_PYTHON = python)
    library(reticulate)
    get_frames <- function() {
      r_stack <- sys.calls()
      py_stack <- py_capture_output(py_run_string("from traceback import print_stack; print_stack()"))
      list(r = r_stack, py = py_stack)
    }
    frames_before <- get_frames()
    py_run_string("print('Initialized Python')")

    py_run_string(glue::trim("
      print('Starting', flush=True)
      try:
        i = 0
        while True:
          i += 1
      except KeyboardInterrupt:
        print('Caught interrupt; ', end='')
      finally:
        print('Running finally; ', end='')

      print('Python finished; ', end = '')
      "))

    # confirm that the python stack was unwound correctly
    frames_after <- get_frames()
    stopifnot(identical(frames_before, frames_after))

    cat("R Finished!")

  })

  p$poll_io(5000)
  expect_identical(p$read_output_lines(1), "Initialized Python")

  p$poll_io(500)
  expect_identical(p$read_output_lines(1), "Starting")

  p$poll_io(500)
  expect_identical(p$read_output(), "")

  p$interrupt()
  p$wait()

  expect_identical(p$get_exit_status(), 0L)
  output <- p$read_all_output()
  expected <- "Caught interrupt; Running finally; Python finished; R Finished!"
  expect_identical(output, expected)

})


test_that("interrupts can be caught by Python while calling R", {
  skip_on_cran()

  p <- callr::r_bg(args = list(python = py_exe()), function(python) {
    Sys.setenv(RETICULATE_PYTHON = python)
    library(reticulate)

    get_frames <- function() {
      r_stack <- sys.calls()
      py_stack <- py_capture_output(py_run_string("from traceback import print_stack; print_stack()"))
      list(r = r_stack, py = py_stack)
    }
    frames_before <- get_frames()

    py_run_string("print('Initialized Python')")

    py$run_forever_r_func <- function() {
      i <- 0
      repeat {
        i <- i + 1
      }
    }

    py_run_string(glue::trim("
      print('Starting', flush=True)
      try:
        run_forever_r_func()
      except KeyboardInterrupt as e:
        print('Caught interrupt; ', end='')
      finally:
        print('Running finally; ', end='')

      print('Python finished; ', end = '')
      "))

    # confirm that the python stack was unwound correctly
    frames_after <- get_frames()
    stopifnot(identical(frames_before, frames_after))

    cat("R Finished!")
  })

  p$poll_io(5000)
  expect_identical(p$read_output_lines(1), "Initialized Python")

  p$poll_io(500)
  expect_identical(p$read_output_lines(1), "Starting")

  p$poll_io(500)
  expect_identical(p$read_output(), "")

  p$interrupt()
  p$wait()

  expect_identical(p$get_exit_status(), 0L)
  expect_identical(
    p$read_all_output(),
    "Caught interrupt; Running finally; Python finished; R Finished!")

})

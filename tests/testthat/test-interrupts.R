
test_that("Running Python scripts can be interrupted", {

  skip_on_cran()

  time <- import("time", convert = TRUE)

  # interrupt this process shortly
  interruptor <- callr::r_bg(args = list(pid = Sys.getpid()), function(pid) {
    Sys.sleep(1)
    system2("kill", c("-s", "INT", pid))
    # ps::ps_interrupt(ps::ps_handle(pid))
  })

  # tell Python to sleep
  before <- Sys.time()
  interrupted <- tryCatch(time$sleep(5), interrupt = identity)
  after <- Sys.time()

  # check that we caught an interrupt
  expect_s3_class(interrupted, "interrupt")

  # check that we took a small amount of time
  diff <- difftime(after, before, units = "secs")
  expect_true(diff < 2)

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

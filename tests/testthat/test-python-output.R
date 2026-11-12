context("output")

capture_test_output <- function(type) {
  sys <- import("sys")
  py_capture_output(type = type, {
    if ("stdout" %in% type)
      sys$stdout$write("out\n");
    if ("stderr" %in% type)
    sys$stderr$write("err\n");
  })
}

test_that("Python streams can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = c("stdout", "stderr")) ,"out\nerr\n")
})

test_that("Python stdout stream can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = "stdout") , "out\n")
})

test_that("Python stderr stream can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = "stderr") , "err\n")
})

test_that("forked children restore fd-backed Python output streams", {
  skip_on_os("windows")
  skip_if_no_python()

  os <- import("os")
  skip_if(!py_has_attr(os, "register_at_fork"))

  p <- callr::r_bg(
    function() {
      library(reticulate)

      py_run_string("
import os
import sys

inherited_stdout = sys.stdout
inherited_stderr = sys.stderr
pid = os.fork()

# os.fork() returns 0 in the child and the child's PID in the parent.
if pid == 0:
    if sys.stdout is inherited_stdout or sys.stderr is inherited_stderr:
        os._exit(1)

    sys.stdout.write('fork child stdout\\n')
    sys.stderr.write('fork child stderr\\n')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

_, status = os.waitpid(pid, 0)
if status != 0:
    raise RuntimeError('forked child inherited remapped output streams')
if sys.stdout is not inherited_stdout or sys.stderr is not inherited_stderr:
    raise RuntimeError('parent output streams were restored')

class StreamProxy:
    def __init__(self, stream):
        self.stream = stream

    def write(self, message):
        return self.stream.write(message)

    def flush(self):
        return self.stream.flush()

custom_stdout = StreamProxy(sys.__stdout__)
custom_stderr = StreamProxy(sys.__stderr__)
sys.stdout = custom_stdout
sys.stderr = custom_stderr
pid = os.fork()

# os.fork() returns 0 in the child and the child's PID in the parent.
if pid == 0:
    if sys.stdout is not custom_stdout or sys.stderr is not custom_stderr:
        os._exit(1)

    sys.stdout.write('custom fork child stdout\\n')
    sys.stderr.write('custom fork child stderr\\n')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

_, status = os.waitpid(pid, 0)
parent_streams_preserved = (
    sys.stdout is custom_stdout and sys.stderr is custom_stderr
)
sys.stdout = inherited_stdout
sys.stderr = inherited_stderr

if status != 0:
    raise RuntimeError('forked child did not preserve custom output streams')
if not parent_streams_preserved:
    raise RuntimeError('parent custom output streams were replaced')
")

      TRUE
    },
    env = c(
      RETICULATE_PYTHON = py_exe(),
      RETICULATE_REMAP_OUTPUT_STREAMS = "1"
    )
  )

  p$wait()
  expect_identical(p$get_exit_status(), 0L)
  expect_true(p$get_result())
  stdout <- p$read_all_output()
  stderr <- p$read_all_error()
  expect_match(stdout, "fork child stdout", fixed = TRUE)
  expect_match(stderr, "fork child stderr", fixed = TRUE)
  expect_match(stdout, "custom fork child stdout", fixed = TRUE)
  expect_match(stderr, "custom fork child stderr", fixed = TRUE)
})

test_that("Python loggers work with py_capture_output", {

  skip_if(py_version() < "3.2")
  skip_on_os("windows")

  output <- py_capture_output({
    logging <- import("logging")
    l <- logging$getLogger("test.logger")
    l$addHandler(logging$StreamHandler())
    l$setLevel("INFO")
    l$info("info")
  })

  expect_equal(output, "info\n")

  l <- logging$getLogger("test.logger2")
  l$addHandler(logging$StreamHandler())
  l$setLevel("INFO")
  output <- py_capture_output(l$info("info"))

  expect_equal(output, "info\n")

})

local_output_logger <- function(name, envir = parent.frame()) {
  sys <- import("sys")
  io <- import("io")
  logging <- import("logging")
  logger <- logging$getLogger(name)
  old_stdout <- sys$stdout
  old_stderr <- sys$stderr
  old_handlers <- logger$handlers
  old_level <- logger$level
  old_propagate <- logger$propagate

  stdout <- io$StringIO()
  stderr <- io$StringIO()
  stdout_handler <- logging$StreamHandler(stdout)
  stderr_handler <- logging$StreamHandler(stderr)
  withr::defer({
    sys$stdout <- old_stdout
    sys$stderr <- old_stderr
    logger$handlers <- old_handlers
    logger$setLevel(old_level)
    logger$propagate <- old_propagate
    stdout_handler$close()
    stderr_handler$close()
  }, envir)

  sys$stdout <- stdout
  sys$stderr <- stderr
  logger$handlers <- list(stdout_handler, stderr_handler)
  logger$setLevel("INFO")
  logger$propagate <- FALSE

  list(logger = logger, stdout = stdout, stderr = stderr,
       stdout_handler = stdout_handler, stderr_handler = stderr_handler)
}

for (logger_name in c("", "reticulate.test.output")) {
  logger_label <- if (nzchar(logger_name)) "named" else "root"

  test_that(paste(logger_label, "logger streams survive repeated captures"), {
    log <- local_output_logger(logger_name)

    expect_equal(py_capture_output(log$logger$info("first")), "first\nfirst\n")
    expect_equal(py_capture_output(log$logger$info("second")), "second\nsecond\n")
    expect_identical(py_id(log$stdout_handler$stream), py_id(log$stdout))
    expect_identical(py_id(log$stderr_handler$stream), py_id(log$stderr))

    log$logger$info("after")
    expect_equal(log$stdout$getvalue(), "after\n")
    expect_equal(log$stderr$getvalue(), "after\n")
  })

  test_that(paste(logger_label, "logger streams survive nested captures"), {
    log <- local_output_logger(logger_name)

    outer <- py_capture_output({
      log$logger$info("before inner")
      inner <- py_capture_output(log$logger$info("inner"))
      log$logger$info("after inner")
    })

    expect_equal(inner, "inner\ninner\n")
    expect_equal(outer, "before inner\nbefore inner\nafter inner\nafter inner\n")
    expect_identical(py_id(log$stdout_handler$stream), py_id(log$stdout))
    expect_identical(py_id(log$stderr_handler$stream), py_id(log$stderr))

    log$logger$info("after outer")
    expect_equal(log$stdout$getvalue(), "after outer\n")
    expect_equal(log$stderr$getvalue(), "after outer\n")
  })

  test_that(paste(logger_label, "logger stream changes survive capture"), {
    log <- local_output_logger(logger_name)
    destination <- import("io")$StringIO()

    output <- py_capture_output({
      log$logger$info("before")
      # setStream() is only available in Python >= 3.7.
      log$stdout_handler$stream <- destination
      log$logger$info("during")
    })

    expect_equal(output, "before\nbefore\nduring\n")
    expect_identical(py_id(log$stdout_handler$stream), py_id(destination))
    expect_identical(py_id(log$stderr_handler$stream), py_id(log$stderr))

    log$logger$info("after")
    expect_equal(destination$getvalue(), "during\nafter\n")
    expect_equal(log$stdout$getvalue(), "")
    expect_equal(log$stderr$getvalue(), "after\n")
  })

  test_that(paste(logger_label, "logger stream changes survive nested captures"), {
    log <- local_output_logger(logger_name)
    destination <- import("io")$StringIO()

    outer <- py_capture_output({
      log$logger$info("outer before")
      inner <- py_capture_output({
        log$logger$info("inner before")
        log$stderr_handler$stream <- destination
        log$logger$info("inner after")
      })
      log$logger$info("outer after")
    })

    expect_equal(inner, "inner before\ninner before\ninner after\n")
    expect_equal(outer, "outer before\nouter before\nouter after\n")
    expect_identical(py_id(log$stdout_handler$stream), py_id(log$stdout))
    expect_identical(py_id(log$stderr_handler$stream), py_id(destination))

    log$logger$info("after")
    expect_equal(destination$getvalue(), "inner after\nouter after\nafter\n")
    expect_equal(log$stdout$getvalue(), "after\n")
    expect_equal(log$stderr$getvalue(), "")
  })

  test_that(paste(logger_label, "logger streams survive selective captures"), {
    log <- local_output_logger(logger_name)

    expect_equal(py_capture_output(type = "stdout", log$logger$info("first")),
                 "first\n")
    expect_equal(log$stderr$getvalue(), "first\n")
    expect_equal(py_capture_output(type = "stderr", log$logger$info("second")),
                 "second\n")
    expect_equal(log$stdout$getvalue(), "second\n")
    expect_identical(py_id(log$stdout_handler$stream), py_id(log$stdout))
    expect_identical(py_id(log$stderr_handler$stream), py_id(log$stderr))

    log$logger$info("after")
    expect_equal(log$stdout$getvalue(), "second\nafter\n")
    expect_equal(log$stderr$getvalue(), "first\nafter\n")
  })

  test_that(paste(logger_label, "logger streams are restored after an error"), {
    log <- local_output_logger(logger_name)

    expect_error(py_capture_output({
      log$logger$info("before error")
      stop("capture error")
    }), "capture error")
    expect_identical(py_id(log$stdout_handler$stream), py_id(log$stdout))
    expect_identical(py_id(log$stderr_handler$stream), py_id(log$stderr))

    log$logger$info("after error")
    expect_equal(log$stdout$getvalue(), "after error\n")
    expect_equal(log$stderr$getvalue(), "after error\n")
  })
}


test_that("nested py_capture_output() calls work", {

  # capture original py ids to check we restored
  # everything correctly at the end
  sys <- import("sys")
  og_sys.stdout_pd_id <- py_id(sys$stdout)
  og_sys.stderr_pd_id <- py_id(sys$stderr)
  og_sys.__stdout___pd_id <- py_id(sys$`__stdout__`)
  og_sys.__stderr___pd_id <- py_id(sys$`__stderr__`)

  # Outer level captures both stdout and stderr
  level_1 <- py_capture_output({

    py_run_string("print('Start outer')")

    # Middle level is configured to only capture stdout,
    # allowing stderr to propagate to the outer level
    level_2 <- py_capture_output(type = "stdout", {

      py_run_string("print('Start middle')")

      # Innermost level captures both stdout and stderr
      level_3 <- py_capture_output({
        py_run_string("print('Start inner')")
        py_run_string("import sys; print('Innermost error', file=sys.stderr)")
        py_run_string("print('End inner')")
      })

      # Middle level generating stderr, should be captured by level_1
      py_run_string("import sys; print('Middle level error', file=sys.stderr)")
      py_run_string("print('End middle')")

    })

    py_run_string("print('End outer')")
  })

  # level_1 captures both stdout and stderr, including the
  # stderr propagated from the middle level
  expect_equal(level_1, "Start outer\nMiddle level error\nEnd outer\n")

  # level_2 only captures stdout, so the stderr from the middle level is not here
  expect_equal(level_2, "Start middle\nEnd middle\n")

  # level 3 captures both stdout and stderr
  expect_equal(level_3, "Start inner\nInnermost error\nEnd inner\n")

  # Check the original streams were restored correctly
  sys <- import("sys")
  expect_identical(og_sys.stdout_pd_id, py_id(sys$stdout))
  expect_identical(og_sys.stderr_pd_id, py_id(sys$stderr))
  expect_identical(og_sys.__stdout___pd_id, py_id(sys$`__stdout__`))
  expect_identical(og_sys.__stderr___pd_id, py_id(sys$`__stderr__`))

})

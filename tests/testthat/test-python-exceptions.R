context("exceptions")


test_that("py_last_error() returns R strings", {
  skip_if_no_python()

  tryCatch(py_eval("range(3)[3]"), error = identity)

  er <- py_last_error()
  expect_identical(er$type, "IndexError")
  expect_type(er$value, "character")
  expect_type(er$traceback, "character")
  expect_type(er$message, "character")

})


test_that("py_last_error() returns the R error condition object", {
    skip_if_no_python()

    signal_simple_error <- function() {
      stop("Hello signal_simple_error")
    }
    raise_py_exception <- function() {
      py_run_string("raise RuntimeError('Hello raise_py_exception')")
    }
    signal_py_exception <- function() {
      ex <- import_builtins()$RuntimeError("Hello signal_py_exception")
      ex$r_call <- sys.call()
      stop(ex)
    }

    f1 <- py_func(signal_simple_error)
    f2 <- py_func(function() f1())
    f3 <- py_func(function() f2())
    f4 <- py_func(function() f3())

    g1 <- py_func(raise_py_exception)
    g2 <- py_func(function() g1())
    g3 <- py_func(function() g2())
    g4 <- py_func(function() g3())

    h1 <- py_func(signal_py_exception)
    h2 <- py_func(function() h1())
    h3 <- py_func(function() h2())
    h4 <- py_func(function() h3())

    for (fn in list(f1, f2, f3, f4,
                    g1, g2, g3, g4,
                    h1, h2, h3, h4)) {

      e <- tryCatch( fn(), error = function(e) e )

      expect_s3_class(e, c("python.builtin.Exception",
                           "python.builtin.BaseException",
                           "python.builtin.object",
                           "error", "condition"))

      expect_identical(conditionMessage(e), e$message)
      expect_identical(conditionCall(e), e$call)

      expect_match(conditionMessage(e), "Hello")
      expect_type(conditionCall(e), "language")
    }

})

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
      stop(import_builtins()$RuntimeError("Hello signal_py_exception"))
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
    # test that py_last_error() reports full r_trace
    # even if python discards the exception object

    catch_and_replace_exception <- py_run_string("
def catch_and_replace_exception(fn):
    try:
      fn()
    except:
      raise RuntimeError('''It's a mystery!''')
")$catch_and_replace_exception

    catch_clear_errstatus_then_raise_new_exception <- py_run_string("
def catch_clear_errstatus_then_raise_new_exception(fn):
    failed = False
    try:
      res = fn()
    except:
      failed = True
    if failed:
      raise RuntimeError('''It's a mystery!''')
    return res
")$catch_clear_errstatus_then_raise_new_exception

    expect_match2 <- expect_match
    formals(expect_match2)$fixed <- TRUE
    formals(expect_match2)$all <- FALSE

    for (erroring_fn in list(signal_simple_error,
                             raise_py_exception,
                             signal_py_exception)) {
      f1 <- py_func(erroring_fn)
      f2 <- py_func(function() f1())
      f3 <- py_func(function() catch_and_replace_exception(f2))
      f4 <- py_func(function() f3())
      f5 <- py_func(function() f4())

      e <- tryCatch(f5(), error = identity)
      output <- suppressMessages(capture.output(print(reticulate::py_last_error())))

      expect_match2(output, "Hello")
      expect_match2(output, "It's a mystery!")
      expect_match2(output, "f1()")
      expect_match2(output, "catch_and_replace_exception(f2)")
      expect_match2(output, "f3()")
      expect_match2(output, "f4()")
      expect_match2(output, "f5()")

      f1 <- py_func(erroring_fn)
      f2 <- py_func(function() f1())
      f3 <- py_func(function() catch_clear_errstatus_then_raise_new_exception(f2))
      f4 <- py_func(function() f3())
      f5 <- py_func(function() f4())

      e <- tryCatch(f5(), error = identity)

      output <- suppressMessages(capture.output(print(reticulate::py_last_error())))

      expect_match2(output, "It's a mystery!") # python code made it a mystery
      expect_match2(output, "Hello") # we make it not a mystery by providing the R trace
      expect_match2(output, "f1()")
      expect_match2(output, "catch_clear_errstatus_then_raise_new_exception(f2)")
      expect_match2(output, "f3()")
      expect_match2(output, "f4()")
      expect_match2(output, "f5()")

    }


})

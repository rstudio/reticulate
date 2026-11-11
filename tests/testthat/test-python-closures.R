context("closures")

test_that("R functions are converted to Python closures", {
  skip_if_no_python()
  func <- function(x, y) x - y
  pyfunc <- test$reflect(func)
  expect_equal(func(10,15), test$callFunc(pyfunc, 10,15))
})

test_that("R functions can accept named arguments from Python", {
  skip_if_no_python()
  func <- function(x, y) x - y
  pyfunc <- test$reflect(func)
  expect_equal(func(x = 15, y = 10), test$callFunc(pyfunc, y = 10, x = 15))
})

test_that("R callbacks receive positional and keyword arguments with either conversion setting", {
  skip_if_no_python()
  invoke <- py_eval("lambda f: f(1, 2.5, label='\\u00fc\\U0001f40d')",
                    convert = FALSE)
  invoke_empty <- py_eval("lambda f: f()", convert = FALSE)

  for (convert in c(TRUE, FALSE)) {
    received <- NULL
    callback <- r_to_py(function(...) {
      received <<- list(...)
      NULL
    }, convert = convert)

    expect_true(py_is_none(invoke(callback)))
    if (!convert) {
      expect_true(all(vapply(received, is_py_object, logical(1))))
      received <- lapply(received, py_to_r)
    }
    expect_identical(received, list(1L, 2.5, label = "\u00fc\U0001f40d"))

    expect_true(py_is_none(invoke_empty(callback)))
    expect_identical(received, list())
  }
})



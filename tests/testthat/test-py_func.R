context("function wrapping")

test_that("R functions can be wrapped in a Python function with the same signature", {
  skip_if_no_python()

  # R function
  f1 <- function(a, b = 3) {
    a + b
  }

  # The same function but re-written in Python
  util <- py_run_string("
def f1(a, b=3):
  return a + b
")

  # signatures should match
  inspect <- import("inspect")
  expect_equal(
    inspect$getargspec(py_func(f1)),
    inspect$getargspec(util$f1))

  # results should match
  expect_equal(
    py_func(f1)(1),
    util$f1(1))
  expect_equal(
    py_func(f1)(1, 2),
    util$f1(1, 2))
  expect_equal(
    py_func(f1)(a = 1, b = 2),
    util$f1(a = 1, b = 2))

  has_args <- function(f) {
    length(inspect$getargspec(f)$args) != 0
  }
  # Some micellaneous test cases which should not fail
  expect_true(has_args(py_func(function(a = c(1, 2, 3)) {})))
  expect_true(has_args(py_func(function(a = NULL) {})))
  expect_true(has_args(py_func(function(a = "abc") {})))
  expect_true(has_args(py_func(function(a, b = 3) {})))
  expect_true(has_args(py_func(function(a = list()) {})))
  expect_true(has_args(py_func(function(a = list("a", 1, list(3, "b", NULL))) {})))
  # TODO: test case for py_func(function(x = NA) {})
  # currently blocked by https://github.com/rstudio/reticulate/issues/197

  # Should error out if the R function's signature
  # contains esoteric Python-incompatible constructs
  expect_error(py_func(function(a = 1, b) {}))
  expect_error(py_func(function(a.b) {}))
})

test_that("R functions wrapped in py_main_thread_func are called on the main thread", {
  skip_if_no_python()
  expect_equal(test$invokeOnThread(py_main_thread_func(function(x) x + 1), 41), 42)
})

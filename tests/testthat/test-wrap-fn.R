context("Function Wrapping")

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
    inspect$getargspec(wrap_fn(f1)),
    inspect$getargspec(util$f1))

  # results should match
  expect_equal(
    wrap_fn(f1)(1),
    util$f1(1))
  expect_equal(
    wrap_fn(f1)(1, 2),
    util$f1(1, 2))
  expect_equal(
    wrap_fn(f1)(a = 1, b = 2),
    util$f1(a = 1, b = 2))
})

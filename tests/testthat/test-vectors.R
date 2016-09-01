context("vectors")

# some helpers
main <- tensorflow:::py_import("__main__")
tensorflow:::py_run_string("def isScalar(x): return not isinstance(x, (list, tuple))")
tensorflow:::py_run_string("def isList(x): return isinstance(x, (list))")
expect_py_true <- function(object) {
  expect_true(tensorflow:::py_to_r(object))
}
expect_py_false <- function(object) {
  expect_false(tensorflow:::py_to_r(object))
}


test_that("Single element vectors are treated as scalars", {
  expect_py_true(main$isScalar(5))
  expect_py_true(main$isScalar(5L))
  expect_py_true(main$isScalar("5"))
  expect_py_true(main$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  expect_py_true(main$isList(c(5,5)))
  expect_py_true(main$isList(c(5L,5L)))
  expect_py_true(main$isList(c("5", "5")))
  expect_py_true(main$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  expect_py_false(main$isScalar(list(5)))
  expect_py_false(main$isScalar(list(5L)))
  expect_py_false(main$isScalar(list("5")))
  expect_py_false(main$isScalar(list(TRUE)))
})

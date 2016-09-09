context("vectors")

# some helpers
main <- tensorflow:::py_module()
tensorflow:::py_run_string("def isScalar(x): return not isinstance(x, (list, tuple))")
tensorflow:::py_run_string("def isList(x): return isinstance(x, (list))")

test_that("Single element vectors are treated as scalars", {
  expect_true(main$isScalar(5))
  expect_true(main$isScalar(5L))
  expect_true(main$isScalar("5"))
  expect_true(main$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  expect_true(main$isList(c(5,5)))
  expect_true(main$isList(c(5L,5L)))
  expect_true(main$isList(c("5", "5")))
  expect_true(main$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  expect_false(main$isScalar(list(5)))
  expect_false(main$isScalar(list(5L)))
  expect_false(main$isScalar(list("5")))
  expect_false(main$isScalar(list(TRUE)))
})

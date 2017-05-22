context("comparisons")

source("utils.R")

test_that("Python integers can be compared", {
  skip_if_no_python()
  a <- py$int(1)
  b <- py$int(2)
  expect_true(a < b)
  expect_true(b > a)
  expect_true(a == a)
  expect_true(a != b)
  expect_true(a <= 1L)
  expect_true(b >= 2L)
})

test_that("Python objects with the same reference are equal", {
  skip_if_no_python()
  list_fn <- py$list
  expect_true(list_fn == py$list)
})



context("comparisons")

source("utils.R")

test_that("Python integers can be compared", {
  skip_if_no_python()
  builtins <- import_builtins(convert = FALSE)
  a <- builtins$int(1)
  b <- builtins$int(2)
  expect_true(a < b)
  expect_true(b > a)
  expect_true(a == a)
  expect_true(a != b)
  expect_true(a <= 1L)
  expect_true(b >= 2L)
})

test_that("Python objects with the same reference are equal", {
  skip_if_no_python()
  builtins <- import_builtins(convert = FALSE)
  list_fn <- builtins$list
  expect_true(list_fn == builtins$list)
})



context("comparisons")

test_that("Python integers can be compared", {
  skip_if_no_python()
  builtins <- import_builtins(convert = FALSE)
  a <- builtins$int(1)
  b <- builtins$int(2)
  expect_true(py_bool(a < b))
  expect_true(py_bool(b > a))
  expect_true(py_bool(a == a))
  expect_true(py_bool(a != b))
  expect_true(py_bool(a <= 1L))
  expect_true(py_bool(b >= 2L))
})

test_that("Python objects with the same reference are equal", {
  skip_if_no_python()
  list_fn1 <- import_builtins(convert = FALSE)$list
  list_fn2 <- import_builtins(convert = TRUE)$list

  expect_true(py_bool(list_fn1 == list_fn2))
})


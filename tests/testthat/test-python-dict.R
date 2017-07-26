context("dict")

source("utils.R")


test_that("Python dictionaries can be created", {
  skip_if_no_python()
  expect_is(dict(), "python.builtin.dict")  
})

test_that("Python dictionaries can use python objects as keys", {
  skip_if_no_python()
  py <- import_builtins(convert = FALSE)
  key <- py$int(42)
  expect_error(dict(key = "foo"), NA)
})

test_that("Python dictionaries have numeric keys", {
  skip_if_no_python()
  expect_error(dict(`42` = "foo"), NA)
})

test_that("Python dictionaries can include numbers in their keys", {
  skip_if_no_python()
  expect_error(dict(foo42 = "foo"), NA)
})

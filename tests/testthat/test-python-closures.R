context("closures")

source("utils.R")

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

test_that("Python function signatures are converted correctly", {
  skip_if_no_python()
  help <- import("rpytools.help")
  func <- function(features, labels) features - labels
  pyfunc <- test$reflect(func)
  expect_equal(help$generate_signature_for_function(func), "(features, labels)")
})


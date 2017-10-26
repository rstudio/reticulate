context("functions")

source("utils.R")

test_that("Python functions are marshalled as function objects", {
  skip_if_no_python()
  spec <- inspect$getargspec(inspect$getclasstree)
  expect_equal(spec$args, c("classes", "unique"))
})

test_that("Python functions can be called by python", {
  skip_if_no_python()
  x <- "foo"
  expect_equal(test$callFunc(test$asString, x), x)
})

test_that("Python callables can be called by R", {
  skip_if_no_python()
  callable <- test$create_callable()
  expect_equal(callable(10), 10)
})

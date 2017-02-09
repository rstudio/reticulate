context("vectors")

source("utils.R")

test_that("Single element vectors are treated as scalars", {
  skip_if_no_python()
  expect_true(test$isScalar(5))
  expect_true(test$isScalar(5L))
  expect_true(test$isScalar("5"))
  expect_true(test$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  skip_if_no_python()
  expect_true(test$isList(c(5,5)))
  expect_true(test$isList(c(5L,5L)))
  expect_true(test$isList(c("5", "5")))
  expect_true(test$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  skip_if_no_python()
  expect_false(test$isScalar(list(5)))
  expect_false(test$isScalar(list(5L)))
  expect_false(test$isScalar(list("5")))
  expect_false(test$isScalar(list(TRUE)))
})

context("vectors")

# some helpers
test <- import("tftools.test")

test_that("Single element vectors are treated as scalars", {
  expect_true(test$isScalar(5))
  expect_true(test$isScalar(5L))
  expect_true(test$isScalar("5"))
  expect_true(test$isScalar(TRUE))
})

test_that("Multi-element vectors are treated as lists", {
  expect_true(test$isList(c(5,5)))
  expect_true(test$isList(c(5L,5L)))
  expect_true(test$isList(c("5", "5")))
  expect_true(test$isList(c(TRUE, TRUE)))
})

test_that("The list function forces single-element vectors to be lists", {
  expect_false(test$isScalar(list(5)))
  expect_false(test$isScalar(list(5L)))
  expect_false(test$isScalar(list("5")))
  expect_false(test$isScalar(list(TRUE)))
})

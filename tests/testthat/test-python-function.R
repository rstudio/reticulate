context("functions")

# some helpers
inspect <- import("inspect")
test <- import("tftools.test")

test_that("Python functions are marshalled as function objects", {
  spec <- inspect$getargspec(inspect$getclasstree)
  expect_equal(spec$args, c("classes", "unique"))
})

test_that("Python functions can be called by python", {
  x <- "foo"
  expect_equal(test$callFunc(test$asString, x), x)
})


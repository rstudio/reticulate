context("iterators")

# some helpers
test <- import("tftools.test")

test_that("Iterators are converted to R lists", {
  rlist <- list("foo", "bar", 42L)
  expect_equal(test$makeIterator(rlist), rlist)
})

test_that("Iterators of uniform types become typed R vectors", {
  intvector <- seq(5)
  expect_equal(test$makeIterator(intvector), intvector)
  numericvector <- c(1.0, 1.5, 2.0, 2.5, 3.0)
  expect_equal(test$makeIterator(numericvector), numericvector)
})

test_that("Generators are converted to R vectors", {
  expect_equal(test$makeGenerator(5) + 1, seq(5))
})

context("iterators")

# some helpers
test <- import("tftools.test")

test_that("Iterators reflect values back", {
  rlist <- list("foo", "bar", 42L)
  expect_equal(iterate(test$makeIterator(rlist)), rlist)
})


test_that("Generators reflect values back", {
  expect_equal(as.integer(iterate(test$makeGenerator(5))) + 1L, seq(5))
})


test_that("Iterators are drained of their values by iteration", {
  iter <- test$makeIterator(c(1:5))
  a <- iterate(iter)
  b <- iterate(iter)
  expect_length(b, 0)
})

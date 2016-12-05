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

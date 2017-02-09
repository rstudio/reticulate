context("iterators")

source("utils.R")

test_that("Iterators reflect values back", {
  skip_if_no_python()
  rlist <- list("foo", "bar", 42L)
  expect_equal(iterate(test$makeIterator(rlist)), rlist)
})


test_that("Generators reflect values back", {
  skip_if_no_python()
  expect_equal(as.integer(iterate(test$makeGenerator(5))) + 1L, seq(5))
})


test_that("Iterators are drained of their values by iteration", {
  skip_if_no_python()
  iter <- test$makeIterator(c(1:5))
  a <- iterate(iter)
  b <- iterate(iter)
  expect_length(b, 0)
})

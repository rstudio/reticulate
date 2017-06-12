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


test_that("infinite iterators can be accessed with iter_next", {
  skip_if_no_python()
  
  # create an infinite generator
  main <- py_run_string("
def infinite_generator():
  n = 0
  while True:
    yield n
    n += 1
")
  it <- main$infinite_generator()
  
  # iterate and stop when i is 10
  while(TRUE) {
    i <- iter_next(it)
    if (i == 10) break
  }
  expect_equal(i, 10)
})

test_that("iter_next returns sentinel value when it completes", {
  skip_if_no_python()
  iter <- test$makeIterator(c(1:5))
  while (TRUE) {
    item <- iter_next(iter, NA)
    if (is.na(item))
      break
  }
  expect_equal(item, NA)
})


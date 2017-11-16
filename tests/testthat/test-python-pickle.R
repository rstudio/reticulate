context("pickle")

source("utils.R")

test_that("Objects can be saved and loaded with pickle", {
  skip_if_no_python()
  x <- dict()
  x$a <- 1
  x$b <- 2
  py_save_object(x, "x.pickle")
  on.exit(unlink("x.pickle"), add = TRUE)
  y <- py_load_object("x.pickle")
  expect_true(x == y)
})


test_that("The cPickle implementation can be used", {
  skip_if_no_python()
  skip_on_cran()
  expect_error({
    x <- dict()
    x$a <- 1
    x$b <- 2
    py_save_object(x, "x.pickle", pickle = "cPickle")
    unlink("x.pickle")
  }, NA)
})
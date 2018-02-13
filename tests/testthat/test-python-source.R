

context("source")

source("utils.R")

test_that("Python scripts can be sourced", {
  skip_if_no_python()
  source_python('script.py')
  expect_equal(add(2, 4), 6)
})

test_that("source_python assigns into the requested environment", {
  skip_if_no_python()
  env <- new.env(parent = emptyenv())
  source_python('script.py', envir = env)
  expect_equal(env$add(2, 4), 6)
})

test_that("source_python respects the convert argument", {
  skip_if_no_python()
  source_python('script.py', convert = FALSE)
  expect_s3_class(add(2, 4), 'python.builtin.object')
})


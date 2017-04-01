context("wrapper")

source("utils.R")

test_that("test_py_function_wrapper() generates wrapper correctly", {
  skip_if_no_python()
  generated_wrapper <- py_function_wrapper("test$test_py_function_wrapper")
  generated_output <- capture.output(generated_wrapper)
  expected_output <- readLines("expected_wrapper.txt")
  expect_equal(generated_output, expected_output)
})

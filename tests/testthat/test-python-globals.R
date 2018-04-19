context("globals")

source("utils.R")

test_that("Interpreter sessions can be saved and loaded with dill", {
  skip_if_no_python()
  
  py_run_string("x = 1")
  py_run_string("y = 1")
  py_run_string("[globals().pop(i) for i in ['x', 'y']]")
  
  test_x <- tryCatch(
    py_run_string("x = x + 1"),
    error = function(e) {
      py_last_error()$value
    }
  )
  test_y <- tryCatch(
    py_run_string("y = y + 1"),
    error = function(e) {
      py_last_error()$value
    }
  )
  expect_equal(test_x, "name 'x' is not defined")
  expect_equal(test_y, "name 'y' is not defined")
})


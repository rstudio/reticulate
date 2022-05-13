context("run")

test_that("Python code can be run as strings", {
  
  # runs code in main module
  result <- py_run_string("x = 1")
  expect_equal(result$x, 1L)
  
  main <- import_main(convert = TRUE)
  expect_equal(main$x, 1L)
  
  # runs code in local dictionary
  result <- py_run_string("x = 42", local = TRUE)
  expect_true(result$x == 42L)
  expect_true(main$x == 1L)
  
})

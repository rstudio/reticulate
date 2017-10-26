context("strings")

source("utils.R")

test_that("Unicode strings are handled by py_str", {
  skip_if_no_python()
  main <- py_run_string("x = u'\xfc'", convert = FALSE)
  expect_equal(py_str(main$x), "ü")
})

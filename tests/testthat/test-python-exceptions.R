context("exceptions")


test_that("py_last_error() returns R strings", {
  skip_if_no_python()

  tryCatch(py_eval("range(3)[3]"), error = identity)

  er <- py_last_error()
  expect_identical(er$type, "IndexError")
  expect_type(er$value, "character")
  expect_type(er$traceback, "character")
  expect_type(er$message, "character")

})

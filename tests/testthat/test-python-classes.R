context("classes")

test_that("Python class variables are accessible via $", {
  skip_if_no_python()
  expect_equal(test$PythonClass$FOO, 1)
})

test_that("Python class members can be called via $", {
  skip_if_no_python()
  expect_equal(test$PythonClass$class_method(), 1)
})

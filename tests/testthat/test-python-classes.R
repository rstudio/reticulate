context("classes")

test_that("Python class variables are accessible via $", {
  skip_if_no_python()
  expect_equal(test$PythonClass$FOO, 1)
})

test_that("Python class members can be called via $", {
  skip_if_no_python()
  expect_equal(test$PythonClass$class_method(), 1)
})


test_that("py_has_method()", {
  Fraction <- import("fractions")$Fraction

  expect_false(py_has_method(Fraction, "as_integer_ratio"))
  expect_true(py_has_method(Fraction(), "as_integer_ratio"))
})

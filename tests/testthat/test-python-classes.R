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
  # in python2, unbound methods are still methods
  # in python3, only bound, user-defined, functions are methods
  skip_if(py_version() < 3)
  Fraction <- import("fractions")$Fraction

  expect_false(py_has_method(Fraction, "conjugate"))
  expect_true(py_has_method(Fraction(), "conjugate"))
})


if(getRversion() >= "4.3.0")
test_that("nameOfClass()", {

  numpy <- import("numpy")
  x <- r_to_py(array(1:3))
  expect_true(inherits(x, numpy$ndarray))

  io <- import("io")
  builtins <- import_builtins()

  with(builtins$open(tempfile(), "wb") %as% f, {
    expect_true(inherits(f, io$BufferedWriter))
    expect_false(f$closed)
  })
  expect_true(f$closed)

})

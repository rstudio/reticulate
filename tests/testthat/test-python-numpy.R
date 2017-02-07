context("numpy")

if (py_available())
  np <- import("numpy")

test_that("R matrixes are converted to numpy ndarray", {
  skip_if_no_python()
  m1 <- matrix(c(1,2,3,4), nrow = 2, ncol = 2)
  m2 <- m1
  expect_equal(np$equal(m1,m2), m1 == m2)
})

test_that("Numpy ndarray is converted to R matrix", {
  skip_if_no_python()
  m1 <- np$matrix(list(c(1,2), c(3,4)))
  expect_equal(m1, matrix(c(1,2,3,4), nrow = 2, ncol = 2, byrow = TRUE))
})

test_that("Multi-dimensional arrays are handled correctly", {
  skip_if_no_python()
  a1 <- array(c(1:8), dim = c(2,2,2))
  expect_equal(-a1, np$negative(a1))
})

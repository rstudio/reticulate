context("numpy")

source("utils.R")

test_that("R matrixes are converted to numpy ndarray", {
  skip_if_no_numpy()
  np <- import("numpy")
  m1 <- matrix(c(1,2,3,4), nrow = 2, ncol = 2)
  m2 <- m1
  expect_equal(np$equal(m1,m2), m1 == m2)
})

test_that("Numpy ndarray is converted to R matrix", {
  skip_if_no_numpy()
  np <- import("numpy")
  m1 <- np$matrix(list(c(1,2), c(3,4)))
  expect_equal(m1, matrix(c(1,2,3,4), nrow = 2, ncol = 2, byrow = TRUE))
})

test_that("Multi-dimensional arrays are handled correctly", {
  skip_if_no_numpy()
  np <- import("numpy")
  a1 <- array(c(1:8), dim = c(2,2,2))
  expect_equal(-a1, np$negative(a1))
})

test_that("Character arrays are handled correctly", {
  skip_if_no_numpy()
  np <- import("numpy")
  a1 <- array(as.character(c(1:8)), dim = c(2,2,2))
  expect_equal(a1, py_to_r(r_to_py(a1)))
})

test_that("Long integer types are converted to R numeric", {
  skip_if_no_numpy()
  np <- import("numpy", convert = FALSE)
  dtypes <- c(np$int64, np$uint32, np$uint64, np$long, np$longlong)
  lapply(dtypes, function(dtype) {
    a1 <- np$array(c(1L:30L), dtype = dtype)
    expect_equal(class(as.vector(py_to_r(a1))), "numeric")
  })
})

test_that("Numpy scalars are converted to R vectors", {
  skip_if_no_numpy()
  np <- import("numpy")
  scalar <- c(1.1)
  expect_equal(as.numeric(np$array(scalar)), scalar)
})


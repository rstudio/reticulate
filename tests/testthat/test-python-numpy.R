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


test_that("np_array creates arrays of the expected type", {
  skip_if_no_numpy()
  np <- import("numpy")
  a <- np_array(c(1L:10L), dtype = "float32")
  expect_equal(py_to_r(a$dtype$name), "float32")
})

test_that("np_array creates arrays of the expected order", {
  skip_if_no_numpy()
  np <- import("numpy")
  a <- np_array(c(1L:8L), dim = c(2,4), dtype = "float32", order = "F")
  expect_true(py_to_r(a$flags$f_contiguous))
})

test_that("np_array can reshape existing numpy arrays", {
  skip_if_no_numpy()
  np <- import("numpy")
  a <- np_array(c(1L:8L), dim = c(2,4), dtype = "float32")
  a <- np_array(a, dim = c(2,2,2))
  expect_equal(py_to_r(a$shape), list(2L, 2L, 2L))
})

test_that("numpy dim and length functions work", {
  skip_if_no_numpy()
  
  test_array <- function(a) {
    expect_equal(dim(a), c(2,4))
    dim(a) <- c(2,2,2)
    expect_equal(dim(a), c(2,2,2))
    expect_equal(length(a), 8)
  }
  
  # test no-convert numpy array
  a <- np_array(c(1L:8L), dim = c(2,4), dtype = "float32")
  test_array(a)
  
  # test convertable numpy array
  np <- import("numpy")
  test_array(r_to_py(matrix(c(1:8), nrow = 2, ncol = 4)))
})


context("numpy")

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
  dtypes <- c(np$int64, np$int32,
              np$uint64, np$uint32)
  lapply(dtypes, function(dtype) {
    a1 <- np$array(c(1L:30L), dtype = dtype)
    a1 <- as.vector(py_to_r(a1))
    expect_true(is.numeric(a1))
    expect_equal(a1, 1:30)
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
  a <- np_array(c(1L:8L), dtype = "float32", order = "F")
  expect_true(py_to_r(a$flags$f_contiguous))
})

test_that("numpy length functions works", {
  skip_if_no_numpy()

  test_array <- function(a) {
    expect_equal(length(a), 8)
  }

  # test no-convert numpy array
  a <- np_array(c(1L:8L), dtype = "float32")
  test_array(a)

  # test convertable numpy array
  np <- import("numpy")
  test_array(r_to_py(matrix(c(1:8), nrow = 2, ncol = 4)))
})

test_that("boolean matrices are converted appropriately", {
  skip_if_no_numpy()

  A <- matrix(TRUE, nrow = 2, ncol = 2)
  expect_equal(A, py_to_r(r_to_py(A)))
})

test_that("numpy string arrays are correctly handled", {
  # https://github.com/rstudio/reticulate/issues/1409
  skip_if_no_numpy()

  np <- import("numpy", convert = FALSE)
  c1 <- np$array(c("1", "2", "3"))
  c2 <- np$array(c("4", "5", "6"))
  x <- py_to_r(np$stack(list(c1, c2), axis = 1L))
  expect_equal(x, matrix(as.character(1:6), ncol = 2))

  # test for more dimensions
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  # test F ordering
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")
  y <- np$asfortranarray(y)
  expect_true(py_to_r(y$flags$f_contiguous))

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  # test for strided views
  x <- np$random$rand(3L, 5L, 4L)
  y <- x$astype("str")

  x <- np$reshape(x, c(5L, 3L, 4L))
  y <- np$reshape(y, c(5L, 3L, 4L))

  a <- py_to_r(x)
  b <- py_to_r(y)
  storage.mode(b) <- "numeric"
  expect_equal(a, b)

  x <- np$array(as.character(1:18))$reshape(c(-1L, 2L))

  fn <- py_eval("lambda x: x[::2]") # another strided view
  expect_equal(fn(x), matrix(c("1", "2",
                               "5", "6",
                               "9", "10",
                               "13", "14",
                               "17", "18"), byrow = TRUE, ncol = 2))
  x <- np$asfortranarray(x)
  expect_equal(fn(x), matrix(c("1", "2",
                               "5", "6",
                               "9", "10",
                               "13", "14",
                               "17", "18"), byrow = TRUE, ncol = 2))

  })

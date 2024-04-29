context("scipy sparse matrix")

library(methods)
library(Matrix)

# https://github.com/r-lib/testthat/issues/1556
if (!inherits("t", "standardGeneric"))
  setGeneric("t")

check_matrix_conversion <- function(r_matrix, python_matrix) {
  # check that the conversion to python works
  expect_true(all(py_to_r(python_matrix$toarray()) == as.matrix(r_matrix)))
  # check that the conversion to r works
  expect_true(all(py_to_r(python_matrix) == r_matrix))
  # check that S3 methods work
  expect_equal(dim(python_matrix), dim(r_matrix))
  expect_equal(length(python_matrix), length(r_matrix))
}

test_that("Conversion to scipy sparse matrix S3 methods behave with null pointers", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  result <- r_to_py(x)
  temp_file <- file.path(tempdir(), "sparse_matrix.rds")
  saveRDS(result, temp_file)
  result <- readRDS(temp_file)

  # check that S3 methods behave with null pointers
  expect_true(is(result, "scipy.sparse.csc.csc_matrix") || is(result, "scipy.sparse._csc.csc_matrix"))
  expect_true(is.null(dim(result)))
  expect_true(length(result) == 0L)
  file.remove(temp_file)
})

test_that("Conversion between Matrix::dgCMatrix and Scipy sparse CSC matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.csc.csc_matrix") || is(result, "scipy.sparse._csc.csc_matrix"))
  expect_true(is(py_to_r(result), "dgCMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between a small Matrix::dgCMatrix and Scipy sparse CSC matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.csc.csc_matrix") || is(result, "scipy.sparse._csc.csc_matrix"))
  expect_true(is(py_to_r(result), "dgCMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between Matrix::dgRMatrix and Scipy sparse CSR matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  x <- as(x, "RsparseMatrix")
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.csr.csr_matrix") || is(result, "scipy.sparse._csr.csr_matrix"))
  expect_true(is(py_to_r(result), "dgRMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between a small Matrix::dgRMatrix and Scipy sparse CSR matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  x <- as(x, "RsparseMatrix")
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.csr.csr_matrix") || is(result, "scipy.sparse._csr.csr_matrix"))
  expect_true(is(py_to_r(result), "dgRMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between Matrix::dgTMatrix and Scipy sparse COO matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  x <- as(x, "TsparseMatrix")
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.coo.coo_matrix") || is(result, "scipy.sparse._coo.coo_matrix"))
  expect_true(is(py_to_r(result), "dgTMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between a small Matrix::dgTMatrix and Scipy sparse COO matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  x <- as(x, "TsparseMatrix")
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.coo.coo_matrix") || is(result, "scipy.sparse._coo.coo_matrix"))
  expect_true(is(py_to_r(result), "dgTMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between Scipy sparse matrices without specific conversion functions works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  result <- r_to_py(x)$tolil()

  # check that we are testing the right classes
  expect_true(is(result, "scipy.sparse.lil.lil_matrix") || is(result, "scipy.sparse._lil.lil_matrix"))
  expect_true(is(py_to_r(result), "dgCMatrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion between R sparse matrices without specific conversion functions works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000

  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  # symmetrize
  x <- x + t(x)
  x <- as(x, "symmetricMatrix")
  result <- r_to_py(x)

  # check that we are testing the right classes
  expect_true(is(x, "dsCMatrix"))
  expect_true(is(result, "scipy.sparse.csc.csc_matrix") || is(result, "scipy.sparse._csc.csc_matrix"))
  check_matrix_conversion(x, result)
})

test_that("Conversion with unsorted values works in csc", {
  skip_on_cran()
  skip_if_no_scipy()

  sp <- import("scipy.sparse", convert = FALSE)

  # Test data
  indices <- c(1L, 0L, 2L, 1L, 0L)
  indptr <- c(0L, 3L, 5L)
  data <- c(2, 1, 3, 5, 4)

  # create csr matrix and try to convert
  mat_py <- sp$csc_matrix(
    tuple(
      np_array(data),
      np_array(indices),
      np_array(indptr),
      convert = FALSE
    ),
    shape = c(3L, 2L)
  )

  mat_py_to_r <- py_to_r(mat_py)

  mat_r <- Matrix::sparseMatrix(
    i = indices + 1,
    p = indptr,
    x = data,
    dims = c(3L, 2L)
  )

  expect_equal(as.matrix(mat_py_to_r), as.matrix(mat_r))
})

test_that("Conversion with unsorted values works in csr", {
  skip_on_cran()
  skip_if_no_scipy()

  sp <- import("scipy.sparse", convert = FALSE)

  # Test data
  indices <- c(1L, 0L, 2L, 1L, 0L)
  indptr <- c(0L, 3L, 5L)
  data <- c(2, 1, 3, 5, 4)

  # create csr matrix and try to convert
  mat_py <- sp$csr_matrix(
    tuple(
      np_array(data),
      np_array(indices),
      np_array(indptr),
      convert = FALSE
    ),
    shape = c(2L, 3L)
  )

  mat_py_to_r <- py_to_r(mat_py)

  mat_r <- Matrix::sparseMatrix(
    j = indices + 1,
    p = indptr,
    x = data,
    dims = c(2L, 3L)
  )

  expect_equal(as.matrix(mat_py_to_r), as.matrix(mat_r))
})


test_that("Conversion with unsorted values works in coo", {
  skip_on_cran()
  skip_if_no_scipy()

  sp <- import("scipy.sparse", convert = FALSE)

  # Test data
  row <- c(1L, 0L, 2L, 1L, 0L)
  col <- c(1L, 0L, 0L, 0L, 1L)
  data <- c(5, 1, 3, 2, 4)

  # create csr matrix and try to convert
  mat_py <- sp$coo_matrix(
    tuple(
      np_array(data),
      tuple(
        np_array(row),
        np_array(col)
      ),
      convert = FALSE
    ),
    shape = c(3L, 2L)
  )

  mat_py_to_r <- py_to_r(mat_py)

  mat_r <- as.matrix(Matrix::sparseMatrix(
    i = row + 1,
    j = col + 1,
    x = data,
    dims = c(3L, 2L)
  ))
  dimnames(mat_r) <- NULL
  expect_equal(as.matrix(mat_py_to_r), mat_r)
})

context("scipy sparse matrix")

library(Matrix)

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
  expect_true(is(result, "scipy.sparse.csc.csc_matrix"))
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
  expect_true(is(result, "scipy.sparse.csc.csc_matrix"))
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
  expect_true(is(result, "scipy.sparse.csr.csr_matrix"))
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
  expect_true(is(result, "scipy.sparse.coo.coo_matrix"))
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
  expect_true(is(result, "scipy.sparse.lil.lil_matrix"))
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
  expect_true(is(result, "scipy.sparse.csc.csc_matrix"))
  check_matrix_conversion(x, result)
})

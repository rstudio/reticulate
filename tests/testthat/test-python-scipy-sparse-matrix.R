context("scipy sparse matrix")

test_that("Conversion between Matrix::dgCMatrix and Scipy sparse matrix works", {
  skip_on_cran()
  skip_if_no_scipy()

  N <- 1000
  x <- sparseMatrix(
    i = sample(N, N),
    j = sample(N, N),
    x = runif(N),
    dims = c(N, N))
  result <- r_to_py(x)
  expect_true(all(py_to_r(result) == x))
  expect_equal(dim(result), dim(x))
  expect_equal(length(result), length(x))
})


context("scipy sparse matrix")

test_that("Conversion between Matrix::dgCMatrix and Scipy sparse matrix works", {
  skip_if_no_scipy()
  
  if (require("Matrix")) {
    N <- 1000
    x <- sparseMatrix(
      i = sample(N, N),
      j = sample(N, N),
      x = runif(N),
      dims = c(N, N))
    result <- r_to_py(x)
    all(py_to_r(result) == x)
  }
})

context("arrays")

expect_reshape <- function(r, dim) {
  expect_equal(
    rearray(r, dim = dim),
    py_to_r(rearray(r_to_py(r), dim = dim))
  )
}

test_that("rearray reshapes R, Python vectors similarily", {
  
  # simple reshaping
  expect_reshape(1:4, c(2, 2))
  expect_reshape(matrix(1:8, nrow = 2), c(2, 2, 2))
  
  # more complicated reshaping
  a <- array(1:20, dim = c(2, 3, 4))
  expect_reshape(a, c(2, 12))
  expect_reshape(a, c(12, 2))
  expect_reshape(a, c(2, 6, 2))
  expect_reshape(a, c(6, 2, 2))
  expect_reshape(a, c(2, 2, 3, 2))
  expect_reshape(a, c(3, 2, 2, 2))
  
})

test_that("rearray and dim<- don't do the same thing", {
  x <- 1:4
  r <- rearray(x, c(2, 2))
  dim(x) <- c(2, 2)
  expect_false(identical(r, x))
})
context("arrays")

expect_reshape <- function(r, dim) {
  expect_equal(
    array_reshape(r, dim = dim),
    py_to_r(array_reshape(r_to_py(r), dim = dim))
  )
}

test_that("rearray reshapes R, Python vectors similarily", {
  skip_if_no_numpy()

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
  skip_if_no_numpy()

  x <- 1:4
  r <- array_reshape(x, c(2, 2))
  dim(x) <- c(2, 2)
  expect_false(identical(r, x))
})



test_that("logical arrays convert correctly", {
  # logical arrays convert to numpy arrays that are
  # strided views of the underlying LGLSXP buffer
  # a little extra testing makes sense.
  skip_if_no_numpy()

  # create logical and integer arrays in R
  larr <- function(...) array(c(T, T, F), c(...))
  iarr <- function(...) array(seq(prod(c(...))), c(...))

  dim <- c(2, 3)

  py_apply_mask <- py_run_string("
def apply_mask(x, mask):
  if not x.flags.writeable:
    x = x.copy()
  x[mask] = 0
  return x
")$apply_mask

  r_apply_mask <- function(x, mask) {
    if(is_py_object(x) && !py_to_r(x$flags$writeable))
      x <- x$copy()
    x[mask] <- 0L
    x
  }

  dim <- c(2, 3, 4, 5)
  for (dim in list(c(3), c(4), c(5),
                   c(2, 2), c(2, 3), c(3, 2),
                   c(2, 2, 2), c(3, 3, 3), c(4, 5, 6), c(6, 5, 4),
                   c(2, 2, 3), c(2, 3, 2), c(3, 2, 2),
                   c(2, 2, 2, 2), c(2, 3, 4, 5))) {

    # Create logical and integer arrays in R
    r_logical_array <- larr(dim)
    r_index_array <- iarr(dim)

    py_logical_array <- r_to_py(r_logical_array)
    py_index_array <- r_to_py(r_index_array)

    # check that round-triping gives an identical array
    expect_identical(r_logical_array, py_to_r(py_logical_array))
    expect_identical(r_index_array, py_to_r(py_index_array))

    r_subset_result <- r_index_array[r_logical_array]
    py_subset_result <- py_index_array[py_logical_array]

    r_from_py_subset_result <- py_to_r(py_subset_result)

    # Check that subsetting works.
    # The results should be equivalent, not identical
    # In numpy boolean array indexing, the search order is row-major, C-style
    # In R logical array indexing, the search order is column-major, Fortran-style
    expect_identical(as.array(r_subset_result),
                     as.array(sort(r_from_py_subset_result)))

    # check that subsetting assignment works identically
    expect_identical(
              py_apply_mask(r_index_array, r_logical_array),
               r_apply_mask(r_index_array, r_logical_array))
    expect_identical(
              py_apply_mask(py_index_array, py_logical_array),
      py_to_r( r_apply_mask(py_index_array, py_logical_array)))

  }

})


test_that("StringDtype arrays convert correctly", {

  np <- import("numpy", convert = FALSE)

  StringDType <- tryCatch(np$dtypes$StringDType, error = function(e) NULL)
  if(is.null(StringDType))
    skip("No NumPy StringDType (numpy<2.0)")

  data <- c("this is a longer string", "short string")

  x <- np$array(data, dtype=StringDType())
  expect_true(startsWith(py_to_r(x$dtype$name), "StringDType"))

  expect_identical(py_to_r(x), array(data))
})

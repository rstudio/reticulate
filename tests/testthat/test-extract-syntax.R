context("extract syntax")

source("utils.R")

grab <- function (x) {
  # evaluate tf object x on the graph
  sess <- tf$Session()
  sess$run(x)
}

arr <- function (...) {
  # create an array with the specified dimensions, and fill it with consecutive
  # increasing integers
  dims <- unlist(list(...))
  array(1:prod(dims), dim = dims)
}

test_that("scalar indexing works", {
  skip_if_no_tensorflow()

  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)
  x3_ <- arr(3, 3, 3)

  # cast to Tensors
  x1 <- tf$constant(x1_)
  x2 <- tf$constant(x2_)
  x3 <- tf$constant(x3_)

  # extract as arrays
  y1_ <- x1_[1]
  y2_ <- x2_[1, 2]
  y3_ <- x3_[1, 2, 3]

  # extract as Tensors
  y1 <- x1[0]
  y2 <- x2[0, 1]
  y3 <- x3[0, 1, 2]

  # they should be equivalent
  expect_equal(y1_, grab(y1))
  expect_equal(y2_, grab(y2))
  expect_equal(y3_, grab(y3))

})

test_that("vector indexing works", {
  skip_if_no_tensorflow()

  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)

  # cast to Tensors
  x1 <- tf$constant(x1_)
  x2 <- tf$constant(x2_)

  # extract as arrays
  y1_ <- x1_[2:3]
  y2_ <- x2_[2:3, 1]

  # extract as Tensors
  y1 <- x1[1:2]
  y2 <- x2[1:2, 0]

  # these should be equivalent (need to coerce R version back to arrays)
  expect_equal(y1_, grab(y1))
  expect_equal(array(y2_), grab(y2))

})

test_that("blank indices retain all elements", {
  skip_if_no_tensorflow()

  # set up arrays
  x1_ <- arr(3)
  x2_ <- arr(3, 3)
  x3_ <- arr(3, 3, 3)
  x4_ <- arr(3, 3, 3, 3)

  # cast to Tensors
  x1 <- tf$constant(x1_)
  x2 <- tf$constant(x2_)
  x3 <- tf$constant(x3_)
  x4 <- tf$constant(x4_)

  # extract as arrays
  y1_ <- x1_[]
  y2_a <- x2_[2:3, ]
  y2_b <- x2_[, 1:2]
  y3_a <- x3_[2:3, 1, ]
  y3_b <- x3_[2:3, , 1]
  y4_ <- x4_[2:3, 1, , 2:3]

  # extract as Tensors
  y1 <- x1[]
  y2a <- x2[1:2, ]  # j missing
  y2b <- x2[, 0:1]
  y3a <- x3[1:2, 0, ]
  y3b <- x3[1:2, , 0]
  y4 <- x4[1:2, 0, , 1:2]

  # these should be equivalent
  expect_equal(y1_, grab(y1))
  expect_equal(y2_a, grab(y2a))
  expect_equal(y2_b, grab(y2b))  #
  expect_equal(y3_a, grab(y3a))
  expect_equal(y3_b, grab(y3b))  #
  expect_equal(y4_, grab(y4))

})

test_that("negative and decreasing indexing errors", {
  skip_if_no_tensorflow()

  # set up Tensors
  x1 <- tf$constant(arr(3))
  x2 <- tf$constant(arr(3, 3))

  # extract with negative indices
  expect_error(x1[-1],
               'negative indexing of Tensors is not curently supported')
  expect_error(x2[1:-2, ],
               'negative indexing of Tensors is not curently supported')
  # extract with decreasing indices
  expect_error(x1[3:2],
               'decreasing indexing of Tensors is not curently supported')
  expect_error(x2[2:1, ],
               'decreasing indexing of Tensors is not curently supported')

})

test_that("too many indices error", {
  skip_if_no_tensorflow()

  # set up Tensor
  x <- tf$constant(arr(3, 3, 3))

  # too many
  expect_error(x[1:2, 2, 0:2, 3],
               'incorrect number of dimensions')
  expect_error(x[1:2, 2, 0:2, 3, , ],
               'incorrect number of dimensions')
  expect_error(x[1:2, 2, 0:2, 3, , drop = TRUE],
               'incorrect number of dimensions')
  # too few
  expect_error(x[],
               'incorrect number of dimensions')
  expect_error(x[1:2, ],
               'incorrect number of dimensions')
  expect_error(x[1:2, 2],
               'incorrect number of dimensions')

})

test_that("silly indices error", {
  skip_if_no_tensorflow()

  # set up Tensor
  x <- tf$constant(arr(3, 3, 3))

  # these should all error and notify the user of the failing index
  expect_error(x[1:2, NULL, 2],
               'index 2 is not numeric and finite')
  # expect_error(x[1:2, NA, 2],
  #              'index 2 is not numeric and finite')  #
  expect_error(x[1:2, Inf, 2],
               'index 2 is not numeric and finite')
  expect_error(x[1:2, 'apple', 2],
               'index 2 is not numeric and finite')
  # expect_error(x[1:2, mean, 2],
  #              'index 2 is not numeric and finite')  #
})

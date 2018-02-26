context("pandas")

test_that("Simple Pandas data frames can be roundtripped", {
  skip_if_no_pandas()
  
  pd <- import("pandas")
  
  before <- iris
  after  <- py_to_r(r_to_py(before))
  mapply(function(lhs, rhs) {
    expect_equal(lhs, rhs)
  }, before, after)
  
})

test_that("Ordered factors are preserved", {
  skip_if_no_pandas()
  pd <- import("pandas")
  
  set.seed(123)
  before <- data.frame(x = ordered(letters, levels = sample(letters)))
  after <- py_to_r(r_to_py(before))
  expect_equal(before, after)
  
})
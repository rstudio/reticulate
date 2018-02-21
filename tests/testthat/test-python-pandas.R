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

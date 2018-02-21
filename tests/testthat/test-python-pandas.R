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

test_that("Dates can be roundtripped", {
  r_to_py(Sys.Date())
  df <- data.frame(x = as.Date(1, origin = "1970-01-01"))
  r_to_py(df$x)
})
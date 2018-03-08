context("datetime")

test_that("R dates can be converted to / from Python datetimes", {
  skip_if_no_numpy()
  
  before <- Sys.Date()
  after <- py_to_r(r_to_py(before))
  
  expect_equal(as.numeric(as.POSIXct(before)), as.numeric(after))
})

test_that("R times can be converted to / from Python datetimes", {
  skip_if_no_numpy()
  
  before <- Sys.time()
  attr(before, "tzone") <- "UTC"
  after <- py_to_r(r_to_py(before))
  
  expect_equal(as.numeric(before), as.numeric(after))
})
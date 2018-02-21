context("dates")

test_that("R dates can be converted to / from Python datetimes", {
  
  before <- Sys.Date()
  after <- py_to_r(r_to_py(before))
  expect_identical(before, after)
  
})
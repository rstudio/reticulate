context("datetime")

test_that("R dates can be converted to / from Python datetimes", {
  
  before <- Sys.Date()
  after <- py_to_r(r_to_py(before))
  expect_identical(before, after)
  
})

test_that("R times can be converted to / from Python datetimes", {
  before <- Sys.time()
  after <- py_to_r(r_to_py(before))
  expect_equal(before, after)
})
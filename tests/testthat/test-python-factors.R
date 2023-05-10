context("factors")

test_that("R factors are converted to character", {
  skip_if_no_python()

  before <- iris$Species
  after <- py_to_r(r_to_py(before))
  expect_equal(as.list(as.character(before)), after)
})


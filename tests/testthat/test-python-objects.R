context("objects")

test_that("python objects with a __setitem__ method can be used", {
  skip_if_no_python()

  library(reticulate)
  py_run_string('
class M:
  def __getitem__(self, k):
    return "M"
')

  m <- py_eval('M()', convert = TRUE)
  expect_equal(m[1], "M")

  m <- py_eval('M()', convert = FALSE)
  expect_equal(m[1], r_to_py("M"))

})

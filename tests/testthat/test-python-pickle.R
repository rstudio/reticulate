context("pickle")

test_that("Objects can be saved and loaded with pickle", {
  skip_if_no_python()
  x <- dict()
  x$a <- 1
  x$b <- 2
  py_save_object(x, "x.pickle")
  on.exit(unlink("x.pickle"), add = TRUE)
  y <- py_load_object("x.pickle")
  expect_identical(y, list(a = 1, b = 2))
  expect_true(py_bool(x == y))
  y <- py_load_object("x.pickle", convert = FALSE)
  expect_s3_class(y, "python.builtin.dict")
  expect_true(py_to_r(x == y))
})


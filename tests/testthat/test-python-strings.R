context("strings")

test_that("py_unicode preserves Unicode strings", {
  skip_if_no_python()
  value <- "\u00fc\U0001f40d"
  converted <- py_unicode(value)
  expect_s3_class(converted, "python.builtin.str")
  expect_identical(py_to_r(converted), value)
})

test_that("Unicode strings are handled by py_str", {
  skip_if_no_python()
  skip_on_cran()
  skip_on_os("windows")
  main <- py_run_string("x = u'\\xfc'", convert = FALSE)
  expect_equal(py_str(main$x), "ü")
})



test_that("subclassed strings convert", {
  skip_if_no_python()
  skip_on_cran()

  # https://github.com/rstudio/reticulate/issues/1348
  # https://github.com/fastai/fastcore/blob/master/fastcore/basics.py#L1015
  PrettyString <- py_run_string(
  'class PrettyString(str):
    def __repr__(self): return self', convert = FALSE)$PrettyString
  expect_identical(py_repr(PrettyString("abc_xyz")),
                   "abc_xyz")
})


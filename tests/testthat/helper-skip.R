
skip_if_no_python <- function() {
  if (!py_available(initialize = TRUE))
    skip("Python bindings not available for testing")
}

skip_if_no_numpy <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_numpy_available())
    skip("NumPy not available for testing")
}

skip_if_no_docutils <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_module_available("docutils"))
    skip("docutils not available for testing.")
}

skip_if_no_pandas <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_module_available("pandas"))
    skip("pandas not available for testing")
}

skip_if_no_scipy <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_module_available("scipy"))
    skip("scipy not available for testing")
  scipy <- import("scipy")
  if (clean_version(scipy$`__version__`) < "1.0")
    skip("scipy version is less than v1.0")
}

skip_if_no_test_environments <- function() {
  skip_on_cran()
  skip_if_no_python()
  skip <- is.na(Sys.getenv("RETICULATE_TEST_ENVIRONMENTS", unset = NA))
  if (skip)
    skip("python environments not available for testing")
}

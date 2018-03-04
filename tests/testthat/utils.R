
# import some modules used by the tests
if (py_available(initialize = TRUE)) {
  test <- import("rpytools.test")
  inspect <- import("inspect") 
  sys <- import("sys")
  builtins <- import_builtins(convert = FALSE)
}

# helper to skip tests if python is not avaialable
skip_if_no_python <- function() {
  if (!py_available(initialize = TRUE))
    skip("Python bindings not available for testing")
}

skip_if_no_numpy <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_numpy_available())
    skip("NumPy not available for testing.")
}

skip_if_no_docutils <- function() {
  skip_on_cran()
  skip_if_no_python()
  if (!py_module_available("docutils"))
    skip("docutils not available for testing.")
}


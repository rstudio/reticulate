
# import some modules used by the tests
if (py_available(initialize = TRUE)) {
  test <- import("rpytools.test")
  inspect <- import("inspect") 
  sys <- import("sys")
  np <- import("numpy")
}

# helper to skip tests if python is not avaialable
skip_if_no_python <- function() {
  if (!py_available(initialize = TRUE))
    skip("Python bindings not available for testing")
}

skip_if_no_numpy <- function() {
  skip_if_no_python()
  if (!py_have_numpy())
    skip("NumPy not available for testing.")
}




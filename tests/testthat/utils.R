
# import some modules used by the tests
if (py_available(initialize = TRUE)) {
  test <- import("rpytools.test")
  inspect <- import("inspect") 
  np <- import("numpy")
}

# helper to skip tests if python is not avaialable
skip_if_no_python <- function() {
  if (!py_available(initialize = TRUE))
    skip("Python bindings not available for testing")
}





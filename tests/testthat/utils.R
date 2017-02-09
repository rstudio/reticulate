
skip_if_no_python <- function() {
  if (!py_available(initialize = TRUE))
    skip("Python bindings not available for testing")
}



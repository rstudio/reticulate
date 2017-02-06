
skip_if_no_python <- function() {
  if (!rpy::py_available())
    skip("Python bindings not available for testing")
}



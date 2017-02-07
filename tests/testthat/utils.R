
skip_if_no_python <- function() {
  if (!reticulate::py_available())
    skip("Python bindings not available for testing")
}



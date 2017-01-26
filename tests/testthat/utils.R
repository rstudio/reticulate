
skip_if_no_python <- function() {
  if (!tensorflow:::py_have_bindings())
    skip("Python bindings not available for testing")
}

skip_if_no_tensorflow <- function() {
  if (is.null(tensorflow::tf))
    skip("TensorFlow not available for testing")
}

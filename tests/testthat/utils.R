
skip_if_no_tensorflow <- function() {
  if (is.null(tensorflow::tf))
    skip("TensorFlow not available for test")
}

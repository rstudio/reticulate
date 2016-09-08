
skip_if_no_tensorflow <- function() {
  if (!is_tensorflow_installed())
    skip("TensorFlow not available for test")
}

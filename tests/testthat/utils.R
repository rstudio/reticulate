
skip_if_no_tensorflow <- function() {
  if (!tensorflow:::is_installed())
    skip("TensorFlow not available for test")
}

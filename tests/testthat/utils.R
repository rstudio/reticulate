
have_tensorflow <- tryCatch({ tensorflow(); TRUE }, error = function(e) FALSE)
skip_if_no_tensorflow <- function() {
  if (!have_tensorflow)
    skip("TensorFlow not available for test")
}

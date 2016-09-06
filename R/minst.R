

#' @export
minst_read_data <- function() {
  input_data <- py_import("tensorflow.examples.tutorials.mnist.input_data")
  flags <- tf$app$flags
  FLAGS <- flags$FLAGS
  flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
  input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)
}

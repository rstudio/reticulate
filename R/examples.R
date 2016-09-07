

#' @export
read_example_data <- function(dataset = c("minst")) {
  dataset <- match.arg(dataset)
  if (dataset == "minst") {
    input_data <- tensorflow("examples.tutorials.mnist.input_data")
    flags <- tensorflow()$app$flags
    FLAGS <- flags$FLAGS
    flags$DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
    input_data$read_data_sets(FLAGS$data_dir, one_hot=TRUE)
  } else {
    stop("Unknown dataset '", dataset, "'")
  }
}

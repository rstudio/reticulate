
#' Obtain a reference to a tensorflow module
#'
#' @param module Name of module (defaults to "tensorflow")
#'
#' @examples
#' \dontrun{
#' tf <- tf_import()
#' input_data <- tf_import("examples.tutorials.mnist.input_data")
#' }
#' @export
tf_import <- function(module = "tensorflow") {
  if (substring(module, 1, 10) != "tensorflow")
    module <- paste("tensorflow", module, sep=".")
  py_import(module)
}

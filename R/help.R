

help_handler <- function(type = c("completion", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    list(package_name = "tensorflow",
         title = NULL,
         signature = "tf$constant(value, dtype=NULL, shape=NULL, name=\"Const\")",
         description = "Creates a constant tensor.")
  } else if (type == "url") {
    help_url_handler.python.object(topic, source)
  }
}

help_url_handler.python.object <- function(topic, source) {
  "https://tensorflow.org"
}

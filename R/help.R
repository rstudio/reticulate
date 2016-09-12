

help_handler <- function(type = c("completion", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    help_completion_handler.python.object(topic, source)
  } else if (type == "url") {
    help_url_handler.python.object(topic, source)
  }
}

help_completion_handler.python.object <- function(topic, source) {
  list(package_name = "tensorflow",
       title = NULL,
       signature = "constant(value, dtype=NULL, shape=NULL, name=\"Const\")",
       description = paste0("Creates a constant tensor. The resulting tensor ",
                            "is populated with values of type dtype, as ",
                            "specified by arguments value and (optionally) ",
                            "shape"),
       args = c("value", "dtype", "shape", "name"),
       arg_descriptions = c(
         "A constant value (or list) of output type dtype.",
         "The type of the elements of the resulting tensor.",
         "Optional dimensions of resulting tensor.",
         "Optional name for the tensor."
       )
  )
}


help_url_handler.python.object <- function(topic, source) {
  "https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op.html#constant"
}

help_formals_handler.python.object <- function(topic, source) {
  list(
    formals = c("value", "dtype", "shape", "name"),
    helpHandler = "tensorflow:::help_handler"
  )
}

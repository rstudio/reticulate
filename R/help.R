

help_handler <- function(type = c("completion", "parameter", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    help_completion_handler.python.object(topic, source)
  } else if (type == "parameter") {
    help_completion_parameter_handler.python.object(source)
  } else if (type == "url") {
    help_url_handler.python.object(topic, source)
  }
}

help_completion_handler.python.object <- function(topic, source) {

  # get a reference to the source
  source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                     error = function(e) NULL)

  if (!is.null(source)) {
    # get the docstring
    inspect <- py_module("inspect")
    doc <- inspect$getdoc(py_get_attr(source, topic))
    if (is.null(doc))
      doc <- ""

    # try to generate a signature
    signature <- NULL
    target <- py_get_attr(source, topic)
    if (py_is_callable(target)) {
      help <- py_module("tftools.help")
      signature <- help$generate_signature_for_function(target)
      if (!is.null(signature))
        signature <- paste0(topic, signature)
    }

    list(title = topic,
         signature = signature,
         description = doc)
  } else {
    NULL
  }
}

help_completion_parameter_handler.python.object <- function(source) {

  # get a reference to the source
  source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                     error = function(e) NULL)

  if (!is.null(source)) {
    list(args = c("value", "dtype", "shape", "name"),
         arg_descriptions = c(
           "A constant value (or list) of output type dtype.",
           "The type of the elements of the resulting tensor.",
           "Optional dimensions of resulting tensor.",
           "Optional name for the tensor.")
    )
  } else {
    NULL
  }
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

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/docs.py

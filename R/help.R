# RStudio IDE custom help handlers

help_handler <- function(type = c("completion", "parameter", "url"), topic, source, ...) {
  type <- match.arg(type)
  if (type == "completion") {
    help_completion_handler.tensorflow.python.object(topic, source)
  } else if (type == "parameter") {
    help_completion_parameter_handler.tensorflow.python.object(source)
  } else if (type == "url") {
    help_url_handler.tensorflow.python.object(topic, source)
  }
}

help_completion_handler.tensorflow.python.object <- function(topic, source) {

  # convert source to object if necessary
  source <- source_as_object(source)
  if (is.null(source))
    return(NULL)

  # use the first paragraph of the docstring as the description
  inspect <- import("inspect")
  description <- inspect$getdoc(py_get_attr(source, topic))
  if (is.null(description))
    description <- ""
  matches <- regexpr(pattern ='\n', description, fixed=TRUE)
  if (matches[[1]] != -1)
    description <- substring(description, 1, matches[[1]])

  # try to generate a signature
  signature <- NULL
  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    signature <- help$generate_signature_for_function(target)
    if (is.null(signature))
      signature <- "()"
    signature <- paste0(topic, signature)
  }

  list(title = topic,
       signature = signature,
       description = description)
}

help_completion_parameter_handler.tensorflow.python.object <- function(source) {

  # split into topic and source
  components <- source_components(source)
  if (is.null(components))
    return(NULL)
  topic <- components$topic
  source <- components$source

  # get the function
  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    args <- help$get_arguments(target)
    if (!is.null(args)) {
      # get the descriptions
      doc <- help$get_doc(target)
      if (is.null(doc))
        arg_descriptions <- args
      else {
        doc <- strsplit(doc, "\n", fixed = TRUE)[[1]]
        arg_descriptions <- sapply(args, function(arg) {
          prefix <- paste0("  ", arg, ": ")
          arg_line <- which(grepl(paste0("^", prefix), doc))
          if (length(arg_line) > 0) {
            arg_description <- substring(doc[[arg_line]], nchar(prefix))
            next_line <- arg_line + 1
            while((arg_line + 1) <= length(doc)) {
              line <- doc[[arg_line + 1]]
              if (grepl("^    ", line)) {
                arg_description <- paste(arg_description, line)
                arg_line <- arg_line + 1
              }
              else
                break
            }
            arg_description <- sub("`None`", "`NULL`", arg_description)
          } else {
            arg
          }
        })
      }
      return(list(
        args = args,
        arg_descriptions = arg_descriptions
      ))
    }
  }

  NULL
}



help_url_handler.tensorflow.python.object <- function(topic, source) {

  # normalize topic and source for various calling scenarios
  if (grepl(" = $", topic)) {
    components <- source_components(source)
    if (is.null(components))
      return(NULL)
    topic <- components$topic
    source <- components$source
  } else {
    source <- source_as_object(source)
    if (is.null(source))
      return(NULL)
  }

  # get help page
  page <- NULL
  inspect <- import("inspect")
  if (inspect$ismodule(source)) {
    module <- paste(source$`__name__`)
    help <- module_help(module, topic)
  } else {
    help <- class_help(class(source), topic)
  }

  if (nzchar(help)) {
    version <- tf$`__version__`
    version <- strsplit(version, ".", fixed = TRUE)[[1]]
    help <- paste0("https://www.tensorflow.org/versions/r",
                   version[1], ".", version[2], "/api_docs/python/",
                   help)
  }

  # return help (can be "")
  help
}


help_formals_handler.tensorflow.python.object <- function(topic, source) {

  target <- py_get_attr(source, topic)
  if (py_is_callable(target)) {
    help <- import("tftools.help")
    args <- help$get_arguments(target)
    if (!is.null(args)) {
      return(list(
        formals = args,
        helpHandler = "tensorflow:::help_handler"
      ))
    }
  }

  # default to NULL if we couldn't get the arguments
  NULL
}

# convert source to object if necessary
source_as_object <- function(source) {

  if (is.character(source)) {
    source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                       error = function(e) NULL)
    if (is.null(source))
      return(NULL)
  }

  source
}

source_components <- function(source) {
  components <- strsplit(source, "\\$")[[1]]
  topic <- components[[length(components)]]
  source <- paste(components[1:(length(components)-1)], collapse = "$")
  source <- source_as_object(source)
  if (!is.null(source))
    list(topic = topic, source = source)
  else
    NULL
}


module_help <- function(module, topic) {

  # do we have a page for this module/topic?
  lookup <- paste(module, topic, sep = ".")
  page <- .module_help_pages[[lookup]]

  # if so then append topic
  if (!is.null(page))
    paste(page, topic, sep = "#")
  else
    ""
}

class_help <- function(class, topic) {

  # call recursively for more than one class
  if (length(class) > 1) {
    # call for each class
    for (i in 1:length(class)) {
      help <- class_help(class[[i]], topic)
      if (nzchar(help))
        return(help)
    }
    # no help found
    return("")
  }

  # do we have a page for this class?
  page <- .class_help_pages[[class]]

  # if so then append class and topic
  if (!is.null(page)) {
    components <- strsplit(class, ".", fixed = TRUE)[[1]]
    class <- components[[length(components)]]
    paste0(page, "#", class, ".", topic)
  } else {
    ""
  }
}

.module_help_pages <- list2env(parent = emptyenv(), list(
  tensorflow.Graph = "framework.html",
  tensorflow.Operation = "framework.html",
  tensorflow.Tensor = "framework.html",
  tensorflow.DType = "framework.html",
  tensorflow.as_dtype = "framework.html",
  tensorflow.device = "framework.html",
  tensorflow.container = "framework.html",
  tensorflow.name_scope = "framework.html",
  tensorflow.control_dependencies = "framework.html",
  tensorflow.convert_to_tensor = "framework.html",
  tensorflow.convert_to_tensor_or_indexed_slices = "framework.html",
  tensorflow.get_default_graph = "framework.html",
  tensorflow.reset_default_graph = "framework.html",
  tensorflow.import_graph_def = "framework.html",
  tensorflow.load_file_system_library = "framework.html",
  tensorflow.load_op_library = "framework.html",
  tensorflow.add_to_collection = "framework.html",
  tensorflow.get_collection = "framework.html",
  tensorflow.get_collection_ref = "framework.html",
  tensorflow.GraphKeys = "framework.html",
  tensorflow.RegisterGradient = "framework.html",
  tensorflow.NoGradient = "framework.html",
  tensorflow.RegisterShape= "framework.html",
  tensorflow.TensorShape = "framework.html",
  tensorflow.Dimension = "framework.html",
  tensorflow.op_scope = "framework.html",
  tensorflow.register_tensor_conversion_function = "framework.html",
  tensorflow.DeviceSpec = "framework.html",
  tensorflow.bytes = "framework.html"
))

.class_help_pages <- list2env(parent = emptyenv(), list(
  tensorflow.python.training.optimizer.Optimizer = "train.html",
  tensorflow.python.framework.tensor_shape.Dimension = "framework.html"
))



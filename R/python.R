
#' Import a Python module
#'
#' Import the specified Python module for calling from R. Use \code{"__main__"}
#' to import the main module.
#'
#' @param module Module name
#' @param silent Return \code{NULL} rather than throwing an error
#'  if the specified module cannot be loaded.
#'
#' @return A Python module
#'
#' @examples
#' \dontrun{
#' main <- import("__main__")
#' sys <- import("sys")
#' }
#'
#' @export
import <- function(module, silent = FALSE) {
  if (silent)
    tryCatch(py_module_impl(module), error = function(e) NULL)
  else
    py_module_impl(module)
}


#' @export
`$.tensorflow.python.object` <- function(x, name) {
  attr <- py_get_attr(x, name)
  if (py_is_callable(attr)) {
    function(...) {
      args <- list()
      keywords <- list()
      dots <- list(...)
      names <- names(dots)
      if (!is.null(names)) {
        for (i in 1:length(dots)) {
          name <- names[[i]]
          if (nzchar(name))
            keywords[[name]] <- dots[[i]]
          else
            args[[length(args) + 1]] <- dots[[i]]
        }
      } else {
        args <- dots
      }
      result = py_call(attr, args, keywords)
      if (is.null(result))
        invisible(result)
      else
        result
    }
  } else {
    py_to_r(attr)
  }
}

#' @export
`[[.tensorflow.python.object` <- `$.tensorflow.python.object`

#' @export
print.tensorflow.python.object <- function(x, ...) {
  if (py_is_null_xptr(x))
    print.default(x)
  else
    py_print(x)
}

#' @export
str.tensorflow.python.object <- function(object, ...) {
  if (py_is_null_xptr(object))
    "<pointer: 0x0>"
  else
    py_str(object)
}

#' @importFrom utils .DollarNames
#' @export
.DollarNames.tensorflow.python.object <- function(x, pattern = "") {

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x))
    return(character())

  # get the names and filter out internal attributes (_*)
  names <- py_list_attributes(x)
  names <- names[substr(names, 1, 1) != '_']

  # get the types
  attr(names, "types") <- py_get_attribute_types(x, names)

  # specify a help_handler
  attr(names, "helpHandler") <- "tensorflow:::help_handler"

  # return
  names
}

#' Create Python dictionary
#'
#' Create a Python dictionary object, including a dictionary whose keys
#' are other Python objects rather than character vectors.
#'
#' @param ... Name/value pairs for dictionary
#'
#' @return A Python dictionary
#'
#' @note
#' This is useful for creating dictionaries keyed by Tensor (required
#' for `\code{feed_dict}` parameters).
#'
#' @export
dict <- function(...) {

  # get the args and their names
  values <- list(...)
  names <- names(values)

  # evaluate names in parent env to get keys
  frame <- parent.frame()
  keys <- lapply(names, function(name) {
    if (exists(name, envir = frame, inherits = TRUE))
      get(name, envir = frame, inherits = TRUE)
    else
      name
  })

  # construct dict
  py_dict(keys, values)
}


#' Evaluate an expression within a context.
#'
#' The \code{with} method for objects of type \code{tensorflow.python.object}
#' implements the context manager protocol used by the Python \code{with}
#' statement. The passed object must implement the
#' \href{https://docs.python.org/2/reference/datamodel.html#context-managers}{context
#' manager} (\code{__enter__} and \code{__exit__} methods.
#'
#' @param data Context to enter and exit
#' @param expr Expression to evaluate within the context
#' @param as Name of variable to assign context to for the duration of the
#'   expression's evaluation (optional).
#' @param ... Unused
#'
#' @export
with.tensorflow.python.object <- function(data, expr, as = NULL, ...) {

  # enter the context
  context <- data$`__enter__`()

  # assign the context if we have an as parameter
  asRestore <- NULL
  if (!missing(as)) {
    as <- deparse(substitute(as))
    as <- gsub("\"", "", as)
    if (exists(as, envir = parent.frame()))
      asRestore <- get(as, envir = parent.frame())
    assign(as, context, envir = parent.frame())
  }

  # evaluate the expression and exit the context
  tryCatch(force(expr),
           finally = {
             data$`__exit__`(NULL, NULL, NULL)
             if (!is.null(asRestore))
               assign(as, asRestore, envir = parent.frame())
           }
          )
}


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

  # get a reference to the source
  source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                     error = function(e) NULL)

  if (!is.null(source)) {
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
  } else {
    NULL
  }
}

help_completion_parameter_handler.tensorflow.python.object <- function(source) {

  # get a reference to the source
  source <- tryCatch(eval(parse(text = source), envir = globalenv()),
                     error = function(e) NULL)

  if (!is.null(source)) {
    list(args = c("value", "dtype", "shape", "name"),
         arg_descriptions = c(
           "A constant value (or list) of output type `dtype`.",
           "The type of the elements of the resulting tensor.",
           "Optional dimensions of resulting tensor.",
           "Optional name for the tensor.")
    )
  } else {
    NULL
  }
}



help_url_handler.tensorflow.python.object <- function(topic, source) {
  "https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op.html#constant"
}

help_formals_handler.tensorflow.python.object <- function(topic, source) {
  list(
    formals = c("value", "dtype", "shape", "name"),
    helpHandler = "tensorflow:::help_handler"
  )
}

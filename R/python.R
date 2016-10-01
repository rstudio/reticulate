
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
print.tensorflow.builtin.object <- function(x, ...) {
  str(x, ...)
}

py_xptr_str <- function(object, expr) {
  if (py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    force(expr)
}

#' @importFrom utils str
#' @export
str.tensorflow.builtin.object <- function(object, ...) {
  if (py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else {
    # call python str method
    str <- py_str(object)

    # pick out class name for case when there is no str method
    match <- regexpr("[A-Z]\\w+ object at ", str)
    if (match != -1)
      str <- gsub(" object at ", "", regmatches(str, match))

    # print str
    cat(str, "\n", sep="")
  }
}

#' @export
`$.tensorflow.builtin.object` <- function(x, name) {
  attrib <- py_get_attr(x, name)
  if (py_is_callable(attrib)) {
    f <- function(...) {
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
      result = py_call(attrib, args, keywords)
      if (is.null(result))
        invisible(result)
      else
        result
    }
    # assign py_object attribute so it marshalls back to python
    # as a native python object
    attr(f, "py_object") <- attrib
    f
  } else {
    py_to_r(attrib)
  }
}

#' @export
`[[.tensorflow.builtin.object` <- `$.tensorflow.builtin.object`


#' @importFrom utils .DollarNames
#' @export
.DollarNames.tensorflow.builtin.object <- function(x, pattern = "") {

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
#' The \code{with} method for objects of type \code{tensorflow.builtin.object}
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
with.tensorflow.builtin.object <- function(data, expr, as = NULL, ...) {

  # enter the context
  context <- data$`__enter__`()

  # check for as and as_envir
  if (!missing(as)) {
    as <- deparse(substitute(as))
    as <- gsub("\"", "", as)
  } else {
    as <- attr(data, "as")
  }
  envir <- attr(data, "as_envir")
  if (is.null(envir))
    envir <- parent.frame()

  # assign the context if we have an as parameter
  asRestore <- NULL
  if (!is.null(as)) {
    if (exists(as, envir = envir))
      asRestore <- get(as, envir = envir)
    assign(as, context, envir = envir)
  }

  # evaluate the expression and exit the context
  tryCatch(force(expr),
           finally = {
             data$`__exit__`(NULL, NULL, NULL)
             if (!is.null(as)) {
               remove(list = as, envir = envir)
               if (!is.null(asRestore))
                 assign(as, asRestore, envir = envir)
             }
           }
          )
}

#' Create local alias for objects in \code{with} statements.
#'
#' @param object Object to alias
#' @param name Alias name
#'
#' @name with-as-operator
#'
#' @keywords internal
#' @export
"%as%" <- function(object, name) {
  as <- deparse(substitute(name))
  as <- gsub("\"", "", as)
  attr(object, "as") <- as
  attr(object, "as_envir") <- parent.frame()
  object
}





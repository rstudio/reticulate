
#' Import a Python module
#'
#' Import the specified Python module for calling from R. If no module
#' name is specified then the \code{__main__} module is imported.
#'
#' @param module Module name
#' @param silent Return \code{NULL} rather than throwing an error
#'  if the specified module cannot be loaded.
#'
#' @return A Python module
#'
#' @export
py_module <- function(module = "__main__", silent = FALSE) {
  if (silent)
    tryCatch(py_module_impl(module), error = function(e) NULL)
  else
    py_module_impl(module)
}


#' @export
`$.python.object` <- function(x, name) {
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
`[[.python.object` <- `$.python.object`

#' @export
print.python.object <- function(x, ...) {
  if (py_is_null_xptr(x))
    print.default(x)
  else
    py_print(x)
}

#' @export
str.python.object <- function(object, ...) {
  if (py_is_null_xptr(object))
    "<pointer: 0x0>"
  else
    py_str(object)
}

#' @importFrom utils .DollarNames
#' @export
.DollarNames.python.object <- function(x, pattern = "") {

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x))
    return(character())

  # get the names and filter out internal attributes (_*)
  names <- py_list_attributes(x)
  names <- names[substr(names, 1, 1) != '_']

  # get the types
  attr(names, "types") <- py_get_attribute_types(x, names)

  # get the doc strings
  inspect <- py_module("inspect")
  attr(names, "docs") <- sapply(names, function(name) {
    inspect$getdoc(py_get_attr(x, name))
  })

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
#' The \code{with} method for objects of type \code{python.object} implements the
#' context manager protocol used by the Python \code{with} statement. The passed
#' object must implement the
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
with.python.object <- function(data, expr, as = NULL, ...) {

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

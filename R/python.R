
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
print.tensorflow.python.object <- function(x, ...) {
  if (py_is_null_xptr(x))
    print.default(x)
  else
    py_print(x)
}

#' @importFrom utils str
#' @export
str.tensorflow.python.object <- function(object, ...) {
  if (py_is_null_xptr(object))
    "<pointer: 0x0>"
  else
    py_str(object)
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


#' Combine values into a list containing a specified type
#'
#' Combine values into a list of a specified type. These functions are similar
#' to the \code{c} function but are geared towards passing lists to
#' Python functions.
#'
#' @param ... Objects to be combined
#'
#' @return A list containing values of the specified type
#'
#' @details
#' When a python API accepts a list of a given type it's often desirable to
#' specify the list as an R literal. Typically this would be done using the
#' \code{c} function, however this approach doesn't deal with lists
#' that contain \code{NULL} (since \code{c} will drop the \code{NULL}) nor
#' with single-element lists (since the R to python conversion layer will
#' treat these as scalars).
#'
#' R APIs also typically convert numeric types behind the scenes as necessary
#' whereas python APIs will sometimes be more strict about types. These
#' functions allow explicit casting to the underlying type required by a
#' Python API (e.g. ensuring that \code{int(1, 2, 3)} is an integer even
#' though R integer literal sytnax was not used).
#'
#' @examples
#' int(1, 2, 3)
#' int(NULL, 42)
#' float(4)
#' bool(NULL, TRUE, FALSE)
#'
#' @name python-list-combine
#' @export
int <- function(...) {
  py_list(as.integer, ...)
}

#' @rdname python-list-combine
#' @export
float <- function(...) {
  py_list(as.numeric, ...)
}

#' @rdname python-list-combine
#' @export
bool <- function(...) {
  py_list(as.logical, ...)
}

# combine the values into an R object that will be marshalled to a python list,
# preserving NULL values and converting them to an underlying type using the
# sepcified converter function
py_list <- function(converter, ...) {
  values <- list(...)
  lapply(values, function(value) {
    if (!is.null(value))
      converter(value)
    else
      NULL
  })
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





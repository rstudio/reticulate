
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

  if (!py_available())
    return(NULL)

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

  # special handling for embedded modules (which don't always show
  # up as "attributes")
  if (py_is_module(x) && !py_has_attr(x, name)) {
    module <- py_get_submodule(x, name)
    if (!is.null(module))
      return(module)
  }

  # default handling
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
            if (is.null(dots[[i]]))
              keywords[name] <- list(NULL)
            else
              keywords[[name]] <- dots[[i]]
          else
            if (is.null(dots[[i]]))
              args[length(args) + 1] <- list(NULL)
            else
              args[[length(args) + 1]] <- dots[[i]]
        }
      } else {
        args <- dots
      }
      result = py_call_object(attrib, args, keywords)
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
  names <- py_suppress_warnings(py_list_attributes(x))
  names <- names[substr(names, 1, 1) != '_']
  names <- sort(names, decreasing = FALSE)

  # get the types
  types <- py_suppress_warnings(py_get_attribute_types(x, names))

  # if this is a module then add submodules
  if (inherits(x, "tensorflow.builtin.module")) {
    name <- x$`__name__`
    if (!is.null(name)) {
      submodules <- sort(py_list_submodules(name), decreasing = FALSE)
      names <- c(names, submodules)
      types <- c(types, rep_len(5L, length(submodules)))
    }
  }

  # set types
  attr(names, "types") <- types

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

#' Create Python tuple
#'
#' Create a Python tuple object
#'
#' @param ... Values for tuple
#'
#' @return A Python tuple
#'
#'
#' @export
tuple <- function(...) {

  # get the args
  values <- list(...)

  # if it's a single value then maybe do some special resolution
  if (length(values) == 1) {

    # alias value
    value <- values[[1]]

    # reflect tuples back
    if (inherits(value, "tensorflow.builtin.tuple"))
      return(value)

    # if it's a list then use the list as the values
    if (is.list(value))
      values <- value
  }

  # construct tuple
  py_tuple(values)
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


#' Traverse a Python iterator or generator
#'
#' @param x Python iterator or generator
#' @param f Function to apply to each item. By default applies the
#'   \code{identity} function which just reflects back the value of the item.
#' @param simplify Should the result be simplified to a vector if possible?
#'
#' @return List or vector containing the results of calling \code{f} on each
#'   item in \code{x} (invisibly).
#'
#' @details Simplification is only attempted all elements are length 1
#'  vectors of type "character", "complex", "double", "integer", or
#'  "logical".
#'
#' @export
iterate <- function(x, f = base::identity, simplify = TRUE) {

  # validate
  if (!inherits(x, "tensorflow.builtin.iterator"))
    stop("iterate function called with non-iterator argument")

  # perform iteration
  result <- py_iterate(x, f)

  # simplify if requested and appropriate
  if (simplify) {

    # attempt to simplify if all elements are length 1
    lengths <- sapply(result, length)
    unique_length <- unique(lengths)
    if (length(unique_length) == 1 && unique_length == 1) {

      # then only simplify if we have a common primitive type
      classes <- sapply(result, class)
      unique_class <- unique(classes)
      if (length(unique_class) == 1 &&
          unique_class %in% c("character", "complex", "double", "integer", "logical")) {
        result <- unlist(result)
      }

    }
  }

  # return invisibly
  invisible(result)
}

#' @export
print.tensorflow.builtin.iterator <- function(x, ...) {
  str(x, ...)
  cat("Python iterator/generator (use tensorflow::iterate to traverse)\n")
}

#' Suppress Python and TensorFlow warnings for an expression
#'
#' @param expr Expression to suppress warnigns for
#'
#' @return Result of evaluating expression
#'
#' @export
py_suppress_warnings <- function(expr) {

  # ignore python warnings
  warnings <- import("warnings")
  warnings$simplefilter("ignore")
  on.exit(warnings$resetwarnings(), add = TRUE)

  # ignore tensorflow warnings
  if (!is.null(tf)) {
    old_verbosity <- tf$logging$get_verbosity()
    tf$logging$set_verbosity(tf$logging$ERROR)
    on.exit(tf$logging$set_verbosity(old_verbosity))
  }

  # evaluate the expression
  force(expr)
}


#' Capture and return Python stdout
#'
#' @param expr Expression to capture stdout for
#'
#' @return Character vector with output
#'
#' @export
py_capture_stdout <- function(expr) {

  output_tools <- import("tftools.output")

  restore <- output_tools$start_stdout_capture()

  on.exit({
    if (!is.null(restore))
      output_tools$end_stdout_capture(restore)
  }, add = TRUE)

  force(expr)

  output <- output_tools$end_stdout_capture(restore)
  restore <- NULL
  output
}

py_is_module <- function(x) {
  inherits(x, "tensorflow.builtin.module")
}

py_get_submodule <- function(x, name) {
  module_name <- paste(py_str(py_get_attr(x, "__name__")), name, sep=".")
  import(module_name, silent = TRUE)
}





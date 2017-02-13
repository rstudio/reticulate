
#' Import a Python module
#'
#' Import the specified Python module for calling from R. Use \code{"__main__"}
#' to import the main module.
#'
#' @param module Module name
#' @param delay_load `TRUE` or a function to delay loading the module until
#'  it is first used (if a function is provided then it will be called 
#'  once the module is loaded). `FALSE` to load the module immediately.
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
import <- function(module, delay_load = FALSE) {
  
  # resolve delay load
  delay_load_function <- NULL
  if (is.function(delay_load)) {
    delay_load_function <- delay_load
    delay_load <- TRUE
  }
  
  # normal case (load immediately)
  if (!delay_load || is_python_initialized()) {
    # ensure that python is initialized (pass top level module as
    # a hint as to which version of python to choose)
    top_level_module <- strsplit(module, ".", fixed = TRUE)[[1]][[1]]
    ensure_python_initialized(required_module = top_level_module)
  
    # import the module
    py_module_import(module)
  }
  
  # delay load case (wait until first access)
  else {
    .globals$delay_load_module <- module
    module_proxy <- new.env(parent = emptyenv())
    module_proxy$module <- module
    if (!is.null(delay_load_function))
      module_proxy$onload <- delay_load_function
    attr(module_proxy, "class") <- c("python.builtin.module", 
                                     "python.builtin.object")
    module_proxy
  }
}

#' @export
print.python.builtin.object <- function(x, ...) {
  str(x, ...)
}


#' Evaluate an expression that prints a string with 
#' a check for a null externalptr
#' 
#' @param x Python object
#' @param expr R expression that uses the object
#'   to print a string.
#'   
#' @details If the object is a null externalptr
#'   then \code{<pointer: 0x0>} will be printed. 
#'   Otherwise the expression will be evaluated.
#' 
#' @export
py_xptr_str <- function(x, expr) {
  if (py_is_null_xptr(x) || !py_available())
    cat("<pointer: 0x0>\n")
  else
    force(expr)
}

#' @importFrom utils str
#' @export
str.python.builtin.object <- function(object, ...) {

  if (py_is_null_xptr(object) || !py_available()) {

    cat("<pointer: 0x0>\n")

  } else {

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
str.python.builtin.module <- function(object, ...) {

  if (py_is_module_proxy(object)) {
    
    cat("Module(", get("module", envir = object), ")\n", sep = "")
    
  } else if (py_is_null_xptr(object) || !py_available()) {

    cat("<pointer: 0x0>\n")

  } else {

    py_xptr_str(object,
                cat("Module(", py_str(py_get_attr(object, "__name__")),
                    ")\n", sep="")
    )
  }
}


#' @export
`$.python.builtin.module` <- function(x, name) {
 
  # resolve module proxies
  if (py_is_module_proxy(x)) 
    py_resolve_module_proxy(x)
  
  `$.python.builtin.object`(x, name)
}


#' @export
`$.python.builtin.object` <- function(x, name) {

  # resolve module proxies
  if (py_is_module_proxy(x)) 
    py_resolve_module_proxy(x)
  
  # skip if this is a NULL xptr
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)

  # special handling for embedded modules (which don't always show
  # up as "attributes")
  if (py_is_module(x) && !py_has_attr(x, name)) {
    module <- py_get_submodule(x, name)
    if (!is.null(module))
      return(module)
  }
  
  # convert 'call' to '__call__' if we aren't masking an underlying 'call'
  if (identical(name, "call") &&  
      !py_has_attr(x, "call") && py_has_attr(x, "__call__")) {
    name <- "__call__"
  }

  # default handling
  attrib <- py_get_attr(x, name)
  if (py_is_callable(attrib)) {
    
    # make an R function
    f <- py_callable_as_function(attrib)
    
    # assign py_object attribute so it marshalls back to python
    # as a native python object
    attr(f, "py_object") <- attrib
    
    # return the function
    f
  } else {
    py_to_r(attrib)
  }
}

#' @export
`[[.python.builtin.object` <- `$.python.builtin.object`


#' @export
.DollarNames.python.builtin.module <- function(x, pattern = "") {
  
  # resolve module proxies (ignore errors since this is occurring during completion)
  result <- tryCatch({
    if (py_is_module_proxy(x)) 
      py_resolve_module_proxy(x)
    TRUE
  }, error = function(e) FALSE)
  if (!result)
    return(character())

  # delegate
  .DollarNames.python.builtin.object(x, pattern)
}

#' @importFrom utils .DollarNames
#' @export
.DollarNames.python.builtin.object <- function(x, pattern = "") {

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x) || !py_available())
    return(character())


  # get the names and filter out internal attributes (_*)
  names <- py_suppress_warnings(py_list_attributes(x))
  names <- names[substr(names, 1, 1) != '_']
  names <- sort(names, decreasing = FALSE)

  # get the types
  types <- py_suppress_warnings(py_get_attribute_types(x, names))

  # if this is a module then add submodules
  if (inherits(x, "python.builtin.module")) {
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
  attr(names, "helpHandler") <- "reticulate:::help_handler"

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
#' for `feed_dict` parameters).
#'
#' @export
dict <- function(...) {

  ensure_python_initialized()

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

  ensure_python_initialized()

  # get the args
  values <- list(...)

  # if it's a single value then maybe do some special resolution
  if (length(values) == 1) {

    # alias value
    value <- values[[1]]

    # reflect tuples back
    if (inherits(value, "python.builtin.tuple"))
      return(value)

    # if it's a list then use the list as the values
    if (is.list(value))
      values <- value
  }

  # construct tuple
  py_tuple(values)
}


#' Convert to Python Unicode Object
#' 
#' @param str Single element character vector to convert
#' 
#' @details By default R character vectors are converted to Python strings.
#'   In Python 3 these values are unicode objects however in Python 2 
#'   they are 8-bit string objects. This function enables you to 
#'   obtain a Python unicode object from an R character vector 
#'   when running under Python 2.
#' 
#' @export
py_unicode <- function(str) {
  ensure_python_initialized()
  py <- import("__builtin__")
  py_call(py_get_attr(py, "unicode"), args = list(str), convert = FALSE)
}



#' Evaluate an expression within a context.
#'
#' The \code{with} method for objects of type \code{python.builtin.object}
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
with.python.builtin.object <- function(data, expr, as = NULL, ...) {

  ensure_python_initialized()

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

#' @export
as.function.python.builtin.object <- function(x, ...) {
  
  ensure_python_initialized()
  
  if (py_is_null_xptr(x))
    stop("Python object is NULL so cannot be convereted to a function")
  else if (py_is_callable(x))
    py_callable_as_function(x)
  else
    stop("Python object is not callable so cannot be converted to a function.")
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

  ensure_python_initialized()

  # validate
  if (!inherits(x, "python.builtin.iterator"))
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
print.python.builtin.iterator <- function(x, ...) {
  str(x, ...)
  cat("Python iterator/generator (use iterate function to traverse)\n")
}

#' Call a Python callable object
#' 
#' @param args List of unnamed arguments
#' @param keywords List of named arguments
#' @param convert `TRUE` to convert the return value to R (otherwise keep it as
#'   a Python object).
#'   
#' @return Return value of call
#'   
#' @keywords internal
#'   
#' @export
py_call <- function(x, args = NULL, keywords = NULL, convert = TRUE) {
  ensure_python_initialized()
  py_call_impl(x, args, keywords, convert)
}


#' Get an attribute of a Python object
#'
#' @param x Python object
#' @param name Attribute name
#' @param silent \code{TRUE} to return \code{NULL} if the attribute
#'  doesn't exist (default is \code{FALSE} which will raise an error)
#'
#' @return Attribute of Python object
#' @export
py_get_attr <- function(x, name, silent = FALSE) {
  ensure_python_initialized()
  py_get_attr_impl(x, name, silent)
}


#' String representation of a Python object
#' 
#' @param x Python object
#' 
#' @return Character vector with result of calling `PyObject_Str` on the object.
#' 
#' @export
py_str <- function(x) {
  ensure_python_initialized()
  py_str_impl(x)
}


#' Suppress Python warnings for an expression
#'
#' @param expr Expression to suppress warnings for
#'
#' @return Result of evaluating expression
#'
#' @export
py_suppress_warnings <- function(expr) {

  ensure_python_initialized()

  # ignore python warnings
  warnings <- import("warnings")
  warnings$simplefilter("ignore")
  on.exit(warnings$resetwarnings(), add = TRUE)

  # ignore other warnings
  contexts <- lapply(.globals$suppress_warnings_handlers, function(handler) {
    handler$suppress()
  })
  on.exit({
    if (length(contexts) > 0) {
      for (i in 1:length(contexts)) {
        handler <- .globals$suppress_warnings_handlers[[i]]
        handler$restore(contexts[[i]])
      }
    }
  }, add = TRUE)
  

  # evaluate the expression
  force(expr)
}


#' Register a handler for calls to py_suppress_warnings
#'
#' @param handler Handler
#'
#' @details Enables packages to register a pair of functions
#'  to be called to suppress and then re-enable warnings
#'
#' @keywords internal
#' @export
register_suppress_warnings_handler <- function(handler) {
  .globals$suppress_warnings_handlers[[length(.globals$suppress_warnings_handlers) + 1]] <- handler
}


#' Capture and return Python output
#'
#' @param expr Expression to capture stdout for
#' @param type Streams to capture (defaults to both stdout and stderr)
#'
#' @return Character vector with output
#'
#' @export
py_capture_output <- function(expr, type = c("stdout", "stderr")) {

  # initialize python if necessary
  ensure_python_initialized()
  
  # resolve type argument
  type <- match.arg(type, several.ok = TRUE)
 
  # get output tools helper functions
  output_tools <- import("rpytools.output")

  # handle stdout
  restore_stdout <- NULL
  if ("stdout" %in% type) {
    restore_stdout <- output_tools$start_stdout_capture()
    on.exit({
      if (!is.null(restore_stdout))
        output_tools$end_stdout_capture(restore_stdout)
    }, add = TRUE)
  }

  # handle stderr
  restore_stderr <- NULL
  if ("stderr" %in% type) {
    restore_stderr <- output_tools$start_stderr_capture()
    on.exit({
      if (!is.null(restore_stderr))
        output_tools$end_stderr_capture(restore_stderr)
    }, add = TRUE)
  }
  
  # evaluate the expression
  force(expr)

  # collect the output
  output <- ""
  if (!is.null(restore_stdout)) {
    output <- paste0(output_tools$end_stdout_capture(restore_stdout), "\n")
    restore_stdout <- NULL
  }
  if (!is.null(restore_stderr)) {
    output <- paste0(output, output_tools$end_stderr_capture(restore_stderr), "\n")
    restore_stderr <- NULL
  }
 
  # return the output
  output
}




#' Run Python code
#'
#' Execute code within the the \code{__main__} Python module.
#'
#' @param code Code to execute
#' @param file File to execute
#'
#' @return Reference to \code{__main__} Python module.
#'
#' @name py_run
#'
#' @export
py_run_string <- function(code) {
  ensure_python_initialized()
  py_run_string_impl(code)
}

#' @rdname py_run
#' @export
py_run_file <- function(file) {
  ensure_python_initialized()
  py_run_file_impl(file)
}


py_callable_as_function <- function(callable) {
  function(...) {
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
    result = py_call(callable, args, keywords)
    if (is.null(result))
      invisible(result)
    else
      result
  }
}


py_is_module <- function(x) {
  inherits(x, "python.builtin.module")
}

py_is_module_proxy <- function(x) {
  inherits(x, "python.builtin.module") && exists("module", envir = x)
}

py_resolve_module_proxy <- function(proxy) {
  
  # name of module to import
  module <- get("module", envir = proxy)
  
  # perform the import -- capture error and ammend it with
  # python configuration information if we have it
  result <- tryCatch(import(module), error = function(e) e)
  if (inherits(result, "error")) {
    message <- paste("Python module", module, "was not found.")
    config <- py_config()
    if (!is.null(config)) {
      message <- paste0(message, "\n\nDetected Python configuration:\n\n",
                        str(config), "\n")
    }
    stop(message, call. = FALSE)
  }
  
  # fixup the proxy 
  py_module_proxy_import(proxy)
  
  # clear the global tracking of delay load modules
  .globals$delay_load_module <- NULL
  
  # call then clear onload if specifed
  if (exists("onload", envir = proxy)) {
    onload <- get("onload", envir = proxy)
    remove("onload", envir = proxy)
    onload()
  }
  
}

py_get_submodule <- function(x, name) {
  module_name <- paste(py_str(py_get_attr(x, "__name__")), name, sep=".")
  result <- tryCatch(import(module_name), error = function(e) e)
  if (inherits(result, "error"))
    NULL
  else
    result
}





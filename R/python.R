
#' Import a Python module
#' 
#' Import the specified Python module for calling from R.
#' 
#' @param module Module name
#' @param as Alias for module name (affects names of R classes)
#' @param path Path to import from
#' @param convert `TRUE` to automatically convert Python objects to their R
#'   equivalent. If you pass `FALSE` you can do manual conversion using the 
#'   [py_to_r()] function.
#' @param delay_load `TRUE` to delay loading the module until it is first used.
#'   `FALSE` to load the module immediately. If a function is provided then it
#'   will be called once the module is loaded. If a list containing `on_load()`
#'   and `on_error(e)` elements is provided then `on_load()` will be called on
#'   successful load and `on_error(e)` if an error occurs.
#'   
#' @details The `import_from_path` function imports a Python module from an
#'   arbitrary filesystem path (the directory of the specified python script is
#'   automatically added to the `sys.path`).
#'   
#' @return A Python module
#'   
#' @examples
#' \dontrun{
#' main <- import_main()
#' sys <- import("sys")
#' }
#' 
#' @export
import <- function(module, as = NULL, convert = TRUE, delay_load = FALSE) {
  
  # if there is an as argument then register a filter for it
  if (!is.null(as)) {
    register_class_filter(function(classes) {
      sub(paste0("^", module), as, classes)
    })
  }
  
  # resolve delay load
  delay_load_environment <- NULL
  delay_load_priority <- 0
  delay_load_functions <- NULL
  if (is.function(delay_load)) {
    delay_load_functions <- list(on_load = delay_load)
    delay_load <- TRUE
  } else if (is.list(delay_load)) {
    delay_load_environment <- delay_load$environment
    delay_load_functions <- delay_load
    if (!is.null(delay_load$priority))
      delay_load_priority <- delay_load$priority
    delay_load <- TRUE
  }
  
  # normal case (load immediately)
  if (!delay_load || is_python_initialized()) {
    # ensure that python is initialized (pass top level module as
    # a hint as to which version of python to choose)
    ensure_python_initialized(required_module = module)
  
    # import the module
    py_module_import(module, convert = convert)
  }
  
  # delay load case (wait until first access)
  else {
    if (is.null(.globals$delay_load_module) || (delay_load_priority > .globals$delay_load_priority)) {
      .globals$delay_load_module <- module
      .globals$delay_load_environment <- delay_load_environment
      .globals$delay_load_priority <- delay_load_priority
    }
    module_proxy <- new.env(parent = emptyenv())
    module_proxy$module <- module
    module_proxy$convert <- convert
    if (!is.null(delay_load_functions)) {
      module_proxy$on_load <- delay_load_functions$on_load
      module_proxy$on_error <- delay_load_functions$on_error
    }
    attr(module_proxy, "class") <- c("python.builtin.module", 
                                     "python.builtin.object")
    module_proxy
  }
}


#' @rdname import
#' @export
import_main <- function(convert = TRUE) {
  ensure_python_initialized()
  import("__main__", convert = convert)
}

#' @rdname import
#' @export
import_builtins <- function(convert = TRUE) {
  ensure_python_initialized()
  if (is_python3())
    import("builtins", convert = convert)
  else
    import("__builtin__", convert = convert)
}


#' @rdname import
#' @export
import_from_path <- function(module, path, convert = TRUE, delay_load = FALSE) {
  
  # add the path to sys.path if it isn't already there
  sys <- import("sys", convert = FALSE)
  if (!path %in% py_to_r(sys$path))
    sys$path$append(path)
  
  # import
  import(module, convert = convert, delay_load = delay_load)
}






#' @export
print.python.builtin.object <- function(x, ...) {
  str(x, ...)
}


#' @importFrom utils str
#' @export
str.python.builtin.object <- function(object, ...) {
  if (!py_available() || py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    cat(py_str(object), "\n", sep="")
}

#' @export
str.python.builtin.module <- function(object, ...) {
  if (py_is_module_proxy(object)) {
    cat("Module(", get("module", envir = object), ")\n", sep = "")
  } else {
    cat(py_str(object), "\n", sep = "")
  } 
}

#' @export
as.character.python.builtin.object <- function(x, ...) {
  py_str(x)
}

#' @export
"==.python.builtin.object" <- function(a, b) {
  py_compare(a, b, "==")
}

#' @export
"!=.python.builtin.object" <- function(a, b) {
  py_compare(a, b, "!=")
}

#' @export
"<.python.builtin.object" <- function(a, b) {
  py_compare(a, b, "<")
}

#' @export
">.python.builtin.object" <- function(a, b) {
  py_compare(a, b, ">")
}

#' @export
">=.python.builtin.object" <- function(a, b) {
  py_compare(a, b, ">=")
}

#' @export
"<=.python.builtin.object" <- function(a, b) {
  py_compare(a, b, "<=")
}


py_compare <- function(a, b, op) {
  ensure_python_initialized()
  py_validate_xptr(a)
  if (!inherits(b, "python.builtin.object"))
    b <- r_to_py(b)
  py_validate_xptr(b)
  py_compare_impl(a, b, op)
}


#' @export
summary.python.builtin.object <- function(object, ...) {
  str(object)
}


#' Convert between Pyton and R objects
#' 
#' @inheritParams import
#' @param x Object to convert
#' 
#' @return Converted object
#' 
#' @name r-py-conversion
#' @export
py_to_r <- function(x) {
  
  ensure_python_initialized()
  
  if (!inherits(x, "python.builtin.object"))
    stop("Object to convert is not a Python object")
  
  py_ref_to_r(x)
}


#' @rdname r-py-conversion
#' @export
r_to_py <- function(x, convert = FALSE) {
  
  ensure_python_initialized()
  
  r_to_py_impl(x, convert = convert)
}


#' @export
`$.python.builtin.module` <- function(x, name) {
 
  # resolve module proxies
  if (py_is_module_proxy(x)) 
    py_resolve_module_proxy(x)
  
  `$.python.builtin.object`(x, name)
}

py_has_convert <- function(x) {
  
  # resolve wrapped environment
  x <- as.environment(x)
  
  # get convert flag 
  if (exists("convert", x, inherits = FALSE))
    get("convert", x, inherits = FALSE)
  else
    TRUE
}

#' @export
`$.python.builtin.object` <- function(x, name) {

  # resolve module proxies
  if (py_is_module_proxy(x)) 
    py_resolve_module_proxy(x)
  
  # skip if this is a NULL xptr
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)

  # deterimine whether this object converts to python
  convert <- py_has_convert(x)
  
  # special handling for embedded modules (which don't always show
  # up as "attributes")
  if (py_is_module(x) && !py_has_attr(x, name)) {
    module <- py_get_submodule(x, name, convert)
    if (!is.null(module))
      return(module)
  }
  
  # get the attrib
  if (is.numeric(name) && (length(name) == 1) && py_has_attr(x, "__getitem__"))
    attrib <- x$`__getitem__`(as.integer(name))
  else if (inherits(x, "python.builtin.dict"))
    attrib <- py_dict_get_item(x, name)
  else
    attrib <- py_get_attr(x, name)
  
  # convert
  if (convert || py_is_callable(attrib)) {
    py_ref_to_r_with_convert(attrib, convert)
  }
  else
    attrib
}



# the as.environment generic enables pytyhon objects that manifest
# as R functions (e.g. for functions, classes, callables, etc.) to 
# be automatically converted to enviroments during the construction
# of PyObjectRef. This makes them a seamless drop-in for standard 
# python objects represented as environments

#' @export
as.environment.python.builtin.object <- function(x) {
  if (is.function(x))
    attr(x, "py_object")
  else
    x
}


#' @export
`[[.python.builtin.object` <- `$.python.builtin.object`


#' @export
`$<-.python.builtin.object` <- function(x, name, value) {
  if (!py_is_null_xptr(x) && py_available())
    py_set_attr(x, name, value)
  else
    stop("Unable to assign value (object reference is NULL)")
  x
}

#' @export
`[[<-.python.builtin.object` <- `$<-.python.builtin.object`


#' @export
`$<-.python.builtin.dict` <- function(x, name, value) {
  if (!py_is_null_xptr(x) && py_available())
    py_dict_set_item(x, name, value)
  else
    stop("Unable to assign value (dict reference is NULL)")
  x
}

#' @export
`[[<-.python.builtin.dict` <- `$<-.python.builtin.dict`

#' @export
length.python.builtin.dict <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else
    py_dict_length(x)
}



#' @export
.DollarNames.python.builtin.module <- function(x, pattern = "") {
  
  # resolve module proxies (ignore errors since this is occurring during completion)
  result <- tryCatch({
    if (py_is_module_proxy(x)) 
      py_resolve_module_proxy(x)
    TRUE
  }, error = clear_error_handler(FALSE))
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

  # check for dictionary
  if (inherits(x, "python.builtin.dict")) {
    
    names <- py_dict_get_keys_as_str(x)
    names <- names[substr(names, 1, 1) != '_']
    types <- rep_len(0L, length(names))
    
  } else {
    # get the names and filter out internal attributes (_*)
    names <- py_suppress_warnings(py_list_attributes(x))
    names <- names[substr(names, 1, 1) != '_']
    names <- sort(names, decreasing = FALSE)
    
    # get the types
    types <- py_suppress_warnings(py_get_attribute_types(x, names))
  }

 
  # if this is a module then add submodules
  if (inherits(x, "python.builtin.module")) {
    name <- py_get_name(x)
    if (!is.null(name)) {
      submodules <- sort(py_list_submodules(name), decreasing = FALSE)
      names <- c(names, submodules)
      types <- c(types, rep_len(5L, length(submodules)))
    }
  }

  if (length(names) > 0) {
    # set types
    attr(names, "types") <- types
  
    # specify a help_handler
    attr(names, "helpHandler") <- "reticulate:::help_handler"
  }
  
  # return
  names
}

#' @export
names.python.builtin.object <- function(x) {
  as.character(.DollarNames(x))
}

#' @export
names.python.builtin.module <- function(x) {
  as.character(.DollarNames(x))
}

#' @export
as.array.numpy.ndarray <- function(x, ...) {
  py_to_r(x)
}

#' @export
as.matrix.numpy.ndarray <- function(x, ...) {
  py_to_r(x)
}

#' @export
as.vector.numpy.ndarray <- function(x, mode = "any") {
  a <- as.array(x)
  as.vector(a, mode = mode)
}

#' @export
as.double.numpy.ndarray <- function(x, ...) {
  a <- as.array(x)
  as.double(a)
}

#' @importFrom graphics plot
#' @export
plot.numpy.ndarray <- function(x, y, ...) {
  plot(as.array(x))
}



#' Create Python dictionary
#' 
#' Create a Python dictionary object, including a dictionary whose keys are 
#' other Python objects rather than character vectors.
#' 
#' @param ... Name/value pairs for dictionary (or a single named list to be 
#'   converted to a dictionary).
#' @param convert `TRUE` to automatically convert Python objects to their R 
#'   equivalent. If you pass `FALSE` you can do manual conversion using the 
#'   [py_to_r()] function.
#'   
#' @return A Python dictionary
#'   
#' @note The returned dictionary will not automatically convert it's elements 
#'   from Python to R. You can do manual converstion with the [py_to_r()] 
#'   function or pass `convert = TRUE` to request automatic conversion.
#'   
#' @export
dict <- function(..., convert = FALSE) {

  ensure_python_initialized()

  # get the args 
  values <- list(...)
  
  # if there is a single element and it's a list then use that
  if (length(values) == 1 && is.list(values[[1]]))
    values <- values[[1]]
  
  # get names
  names <- names(values)

  # evaluate names in parent env to get keys
  frame <- parent.frame()
  keys <- lapply(names, function(name) {
    # allow python objects to serve as keys
    if (exists(name, envir = frame, inherits = TRUE)) {
      key <- get(name, envir = frame, inherits = TRUE)
      if (inherits(key, "python.builtin.object"))
        key
      else
        name
    } else {
      if (grepl("^[0-9]+$", name))
        name <- as.integer(name)
      else
        name
    }
  })
  


  # construct dict
  py_dict(keys, values, convert = convert)
}

#' Create Python tuple
#' 
#' Create a Python tuple object
#' 
#' @inheritParams dict
#' @param ... Values for tuple (or a single list to be converted to a tuple).
#'   
#' @return A Python tuple
#' @note The returned tuple will not automatically convert it's elements from
#'   Python to R. You can do manual converstion with the [py_to_r()] function or
#'   pass `convert = TRUE` to request automatic conversion.
#'   
#' @export
tuple <- function(..., convert = FALSE) {

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
  py_tuple(values, convert = convert)
}

#' @export
length.python.builtin.tuple <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else
    py_tuple_length(x)
}


#' Convert to Python Unicode Object
#' 
#' @param str Single element character vector to convert
#' 
#' @details By default R character vectors are converted to Python strings.
#'   In Python 3 these values are unicode objects however in Python 2 
#'   they are 8-bit string objects. This function enables you to 
#'   obtain a Python unicode object from an R character vector 
#'   when running under Python 2 (under Python 3 a standard Python 
#'   string object is returend).
#' 
#' @export
py_unicode <- function(str) {
  ensure_python_initialized()
  if (is_python3()) {
    r_to_py(str)
  } else {
    py <- import_builtins()
    py_call(py_get_attr(py, "unicode"), str)
  }
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
#' @param it Python iterator or generator
#' @param f Function to apply to each item. By default applies the
#'   \code{identity} function which just reflects back the value of the item.
#' @param simplify Should the result be simplified to a vector if possible?
#' @param completed Sentinel value to return from `iter_next()` if the iteration
#'   completes (defaults to `NULL` but can be any R value you specify).
#'
#' @return For `iterate()`, A list or vector containing the results of calling
#'   \code{f} on each item in \code{x} (invisibly); For `iter_next()`, the next
#'   value in the iteration (or the sentinel `completed` value if the iteration
#'   is complete).
#'
#' @details Simplification is only attempted all elements are length 1 vectors
#'   of type "character", "complex", "double", "integer", or "logical".
#'
#' @export
iterate <- function(it, f = base::identity, simplify = TRUE) {

  ensure_python_initialized()

  # validate
  if (!inherits(it, "python.builtin.iterator"))
    stop("iterate function called with non-iterator argument")

  # perform iteration
  result <- py_iterate(it, f)

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




#' @rdname iterate
#' @export
iter_next <- function(it, completed = NULL) {
  
  # validate
  if (!inherits(it, "python.builtin.iterator"))
    stop("iter_next function called with non-iterator argument")
  
  # call iterator
  py_iter_next(it, completed)

}

#' Call a Python callable object
#' 
#' @param ... Arguments to function (named and/or unnamed)
#'   
#' @return Return value of call as a Python object.
#' 
#' @keywords internal
#'   
#' @export
py_call <- function(x, ...) {
  ensure_python_initialized()
  dots <- py_resolve_dots(list(...))
  py_call_impl(x, dots$args, dots$keywords)
}


#' Check if a Python object has an attribute
#'
#' Check whether a Python object \code{x} has an attribute
#' \code{name}.
#' 
#' @param x A python object.
#' @param name The attribute to be accessed.
#'
#' @return \code{TRUE} if the object has the attribute \code{name}, and
#'   \code{FALSE} otherwise.
#' @export
py_has_attr <- function(x, name) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_has_attr_impl(x, name)
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
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_get_attr_impl(x, name, silent)
}

#' Set an attribute of a Python object
#'
#' @param x Python object
#' @param name Attribute name
#' @param value Attribute value
#'
#' @export
py_set_attr <- function(x, name, value) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_set_attr_impl(x, name, value)
}

#' List all attributes of a Python object
#' 
#' 
#' @param x Python object
#' 
#' @return Character vector of attributes
#' @export
py_list_attributes <- function(x) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_list_attributes_impl(x)
}


#' An S3 method for getting the string representation of a Python object
#' 
#' @param object Python object
#' @param ... Unused
#' 
#' @return Character vector 
#' 
#' @details The default implementation will call `PyObject_Str` on the object.
#' 
#' @export
py_str <- function(object, ...) {
  if (!inherits(object, "python.builtin.object"))
    py_str.default(object)
  else if (py_is_null_xptr(object) || !py_available())
    "<pointer: 0x0>"
  else
    UseMethod("py_str")
}

#' @export
py_str.default <- function(object, ...) {
  "<not a python object>"
}


#' @export
py_str.python.builtin.object <- function(object, ...) {
  
  # get default rep
  str <- py_str_impl(object)
  
  # pick out class name for cases where there is python str method
  match <- regexpr("[A-Z]\\w+ object at ", str)
  if (match != -1)
    str <- gsub(" object at ", "", regmatches(str, match))
  
  # return
  str
}

#' @export
py_str.python.builtin.module <- function(object, ...) {
  paste0("Module(", py_get_name(object), ")")
}

#' @export
py_str.python.builtin.list <- function(object, ...) {
  py_collection_str("List", object)
}

#' @export
py_str.python.builtin.dict <- function(object, ...) {
  py_collection_str("Dict", object)
}

#' @export
py_str.python.builtin.tuple <- function(object, ...) {
  py_collection_str("Tuple", object)
}

py_collection_str <- function(name, object) {
  len <- py_collection_len(object)
  if (len > 10)
    paste0(name, " (", len, " items)")
  else
    py_str.python.builtin.object(object)
}

py_collection_len <- function(object) {
  # do this dance so we can call __len__ on dictionaries (which
  # otherwise overload the $)
  len <- py_get_attr(object, "__len__")
  py_to_r(py_call(len))
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

  # ignore any registered warning output types (e.g. tf warnings)
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
  
  # evaluate while ignoring python warnings
  warnings <- import("warnings")
  with(warnings$catch_warnings(), expr)
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

#' Register a filter for class names
#' 
#' @param filter Function which takes a class name and maps it to an alternate
#'   name
#'   
#' @keywords internal
#' @export
register_class_filter <- function(filter) {
  .globals$class_filters[[length(.globals$class_filters) + 1]] <- filter
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
    std_out <- output_tools$end_stdout_capture(restore_stdout)
    output <- paste0(output, std_out)
    if (nzchar(std_out))
      output <- paste0(output, "\n")
    restore_stdout <- NULL
  }
  if (!is.null(restore_stderr)) {
    std_err <- output_tools$end_stderr_capture(restore_stderr)
    output <- paste0(output, std_err)
    if (nzchar(std_err))
      output <- paste0(output, "\n")
    restore_stderr <- NULL
  }
 
  # return the output
  output
}




#' Run Python code
#' 
#' Execute code within the the \code{__main__} Python module.
#' 
#' @inheritParams import
#' @param code Code to execute
#' @param file File to execute
#' @param local Whether to create objects in a local/private namespace (if
#'   `FALSE`, objects are created within the main module).
#'   
#' @return For `py_eval()`, the result of evaluating the expression; For 
#'   `py_run_string()` and `py_run_file()`, the dictionary associated with 
#'   the code execution.
#'   
#' @name py_run
#'   
#' @export
py_run_string <- function(code, local = FALSE, convert = TRUE) {
  ensure_python_initialized()
  invisible(py_run_string_impl(code, local, convert))
}

#' @rdname py_run
#' @export
py_run_file <- function(file, local = FALSE, convert = TRUE) {
  ensure_python_initialized()
  invisible(py_run_file_impl(file, local, convert))
}

#' @rdname py_run
#' @export
py_eval <- function(code, convert = TRUE) {
  ensure_python_initialized()
  py_eval_impl(code, convert)
}

py_callable_as_function <- function(callable, convert) {
  function(...) {
    dots <- py_resolve_dots(list(...))
    result <- py_call_impl(callable, dots$args, dots$keywords)
    if (convert) {
      result <- py_to_r(result)
      if (is.null(result))
        invisible(result)
      else
        result
    }
    else {
      result
    }
  }
}

py_resolve_dots <- function(dots) {
  args <- list()
  keywords <- list()
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
  list(
    args = args,
    keywords = keywords
  )
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
  
  # collect on_load and on_error
  collect_value <- function(name) {
    if (exists(name, envir = proxy, inherits = FALSE)) {
      value <- get(name, envir = proxy, inherits = FALSE)
      remove(list = name, envir = proxy)
      value
    } else {
      NULL
    }
  }
  on_load <- collect_value("on_load")
  on_error <- collect_value("on_error")
  
  # perform the import -- capture error and ammend it with
  # python configuration information if we have it
  result <- tryCatch(import(module), error = clear_error_handler())
  if (inherits(result, "error")) {
    if (!is.null(on_error)) {
      
      # call custom error handler
      on_error(result)
      
      # error handler can and should call `stop`, this is just a failsafe
      stop("Error loading Python module ", module, call. = FALSE)
      
    } else {
      
      # default error message/handler
      message <- py_config_error_message(paste("Python module", module, "was not found."))
      stop(message, call. = FALSE)
    }
  }
  
  # fixup the proxy 
  py_module_proxy_import(proxy)
  
  # clear the global tracking of delay load modules
  .globals$delay_load_module <- NULL
  .globals$delay_load_environment <- NULL
  .globals$delay_load_priority <- 0
  
  # call on_load if specifed
  if (!is.null(on_load))
    on_load()
}

py_get_name <- function(x) {
  py_to_r(py_get_attr(x, "__name__"))
}

py_get_submodule <- function(x, name, convert = TRUE) {
  module_name <- paste(py_get_name(x), name, sep=".")
  result <- tryCatch(import(module_name, convert = convert), 
                     error = clear_error_handler())
  if (inherits(result, "error"))
    NULL
  else
    result
}

py_filter_classes <- function(classes) {
  for (filter in .globals$class_filters)
    classes <- filter(classes)
  classes
}




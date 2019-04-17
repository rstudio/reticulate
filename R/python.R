
#' Import a Python module
#'
#' Import the specified Python module for calling from R.
#'
#' @param module Module name
#' @param as Alias for module name (affects names of R classes). Note that
#'  this is an advanced parameter that should generally only be used
#'  in package development (since it affects the S3 name of the imported
#'  class and can therefore interfere with S3 method dispatching).
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
      module_proxy$get_module <- delay_load_functions$get_module
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
import_from_path <- function(module, path = ".", convert = TRUE) {

  # normalize path
  path <- normalizePath(path)

  # add the path to sys.path if it isn't already there
  sys <- import("sys", convert = FALSE)
  if (!path %in% py_to_r(sys$path))
    sys$path$append(path)

  # import
  import(module, convert = convert)
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

#' Convert Python bytes to an R character vector
#'
#' @inheritParams base::as.character
#'
#' @param encoding Encoding to use for conversion (defaults to utf-8)
#' @param errors Policy for handling conversion errors. Default is 'strict'
#'  which raises an error. Other possible values are 'ignore' and 'replace'
#'
#' @export
as.character.python.builtin.bytes <- function(x, encoding = "utf-8", errors = "strict", ...) {
  x$decode(encoding = encoding, errors = errors)
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

py_maybe_convert <- function(x, convert) {
  if (convert || py_is_callable(x)) {

    # capture previous convert for attr
    attrib_convert <- py_has_convert(x)

    # temporarily change convert so we can call py_to_r and get S3 dispatch
    envir <- as.environment(x)
    assign("convert", convert, envir = envir)
    on.exit(assign("convert", attrib_convert, envir = envir), add = TRUE)

    # call py_to_r
    x <- py_to_r(x)
  }

  x
}

# helper function for accessing attributes or items from a
# Python object, after validating that we do indeed have
# a valid Python object reference
py_get_attr_or_item <- function(x, name, prefer_attr) {
  # resolve module proxies
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)

  # skip if this is a NULL xptr
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)

  # special handling for embedded modules (which don't always show
  # up as "attributes")
  if (py_is_module(x) && !py_has_attr(x, name)) {
    module <- py_get_submodule(x, name, py_has_convert(x))
    if (!is.null(module))
      return(module)
  }

  # re-cast numeric values as integers
  if (is.numeric(name))
    name <- as.integer(name)

  # attributes must always be indexed by strings, so if
  # we receive a non-string 'name', we call py_get_item
  if (!is.character(name)) {
    item <- py_get_item(x, name)
    return(py_maybe_convert(item, py_has_convert(x)))
  }

  # get the attrib and convert as needed
  if (prefer_attr) {
    object <- py_get_attr(x, name)
  } else {

    # if we have an attribute, attempt to get the item
    # but allow for fallback to that attribute
    if (py_has_attr(x, name)) {
      object <- py_get_item(x, name, silent = TRUE)
      if (is.null(object))
        object <- py_get_attr(x, name)
    } else {
      # we don't have an attribute; only attempt item
      # access and allow normal error propagation
      object <- py_get_item(x, name)
    }

  }

  py_maybe_convert(object, py_has_convert(x))
}

#' @export
`$.python.builtin.object` <- function(x, name) {
  py_get_attr_or_item(x, name, TRUE)
}

#' @export
`[.python.builtin.object` <- function(x, name) {
  py_get_attr_or_item(x, name, FALSE)
}

#' @export
`[[.python.builtin.object` <- function(x, name) {
  py_get_attr_or_item(x, name, FALSE)
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
    Encoding(names) <- "UTF-8"
    types <- rep_len(0L, length(names))

  } else {
    # get the names and filter out internal attributes (_*)
    names <- py_suppress_warnings(py_list_attributes(x))
    names <- names[substr(names, 1, 1) != '_']
    # replace function with `function`
    names <- sub("^function$", "`function`", names)
    names <- sort(names, decreasing = FALSE)

    # get the types
    types <- py_suppress_warnings(py_get_attribute_types(x, names))
  }


  # if this is a module then add submodules
  if (inherits(x, "python.builtin.module")) {
    name <- py_get_name(x)
    if (!is.null(name)) {
      submodules <- sort(py_list_submodules(name), decreasing = FALSE)
      Encoding(submodules) <- "UTF-8"
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
#' @param keys Keys to dictionary (can be Python objects)
#' @param values Values for dictionary
#' @param convert `TRUE` to automatically convert Python objects to their R
#'   equivalent. If you pass `FALSE` you can do manual conversion using the
#'   [py_to_r()] function.
#'
#' @return A Python dictionary
#'
#' @note The returned dictionary will not automatically convert its elements
#'   from Python to R. You can do manual converstion with the [py_to_r()]
#'   function or pass `convert = TRUE` to request automatic conversion.
#'
#' @export
dict <- function(..., convert = FALSE) {

  ensure_python_initialized()

  # get the args
  values <- list(...)

  # flag indicating whether we should scan the parent frame for python
  # objects that should serve as the key (e.g. a Tensor)
  scan_parent_frame <- TRUE

  # if there is a single element and it's a list then use that
  if (length(values) == 1 && is.null(names(values)) && is.list(values[[1]])) {
    values <- values[[1]]
    scan_parent_frame <- FALSE
  }

  # get names
  names <- names(values)

  # evaluate names in parent env to get keys
  frame <- parent.frame()
  keys <- lapply(names, function(name) {
    # allow python objects to serve as keys
    if (scan_parent_frame && exists(name, envir = frame, inherits = TRUE)) {
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
  py_dict_impl(keys, values, convert = convert)
}

#' @rdname dict
#' @export
py_dict <- function(keys, values, convert = FALSE) {
  ensure_python_initialized()
  py_dict_impl(keys, values, convert = convert)
}

#' Create Python tuple
#'
#' Create a Python tuple object
#'
#' @inheritParams dict
#' @param ... Values for tuple (or a single list to be converted to a tuple).
#'
#' @return A Python tuple
#' @note The returned tuple will not automatically convert its elements from
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

#' Length of Python object
#'
#' Get the length of a Python object (equivalent to the Python `len()`
#' built in function).
#'
#' @param x Python object
#'
#' @return Length as integer
#'
#' @export
py_len <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else
    as_r_value(x$`__len__`())
}

#' @export
length.python.builtin.list <- function(x) {
  py_len(x)
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
  if (!is.null(as)) {
    assign(as, context, envir = envir)
  }

  # evaluate the expression and exit the context
  tryCatch(force(expr),
           finally = {
             data$`__exit__`(NULL, NULL, NULL)
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
#' @param x Python iterator or iterable
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

  # resolve iterator
  it <- as_iterator(it)

  # perform iteration
  result <- flowery::drain(it)

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

  if (inherits(it, "iterator"))
    return(it())

  # check for iterator
  if (!inherits(it, "python.builtin.iterator") &&
      !inherits(it, "python.builtin.listiterator"))
    stop("iterator function called with non-iterator argument", call. = FALSE)

  # call iterator
  py_iter_next(it, completed)
}


#' @rdname iterate
#' @export
as_iterator <- function(x) {
  if (inherits(x, "python.builtin.iterator"))
    as_flowery_iterator(x)
  else if (py_has_attr(x, "__iter__"))
    as_flowery_iterator(x$`__iter__`())
  else
    stop("iterator function called with non-iterator argument", call. = FALSE)
}

as_flowery_iterator <- function(x) {
  flowery::generator({
    nxt <- iter_next(x)
    while(!is.null(nxt)) {
      yield(nxt)
      nxt <- iter_next(x)
    }
  })
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
  attrs <- py_list_attributes_impl(x)
  Encoding(attrs) <- "UTF-8"
  attrs
}



#' Get an item from a Python object
#'
#' Retrieve an item from a Python object, similar to how
#' \code{x[name]} might be used in Python code to access an
#' item indexed by `key` on an object `x`. The object's
#' `__getitem__` method will be called.
#'
#' @param x A Python object.
#' @param key The key used for item lookup.
#' @param silent Boolean; when \code{TRUE}, attempts to access
#'   missing items will return \code{NULL} rather than
#'   throw an error.
#'
#' @family item-related APIs
#' @export
py_get_item <- function(x, key, silent = FALSE) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)

  if (!py_has_attr(x, "__getitem__"))
    stop("Python object has no '__getitem__' method", call. = FALSE)
  getitem <- py_to_r(py_get_attr(x, "__getitem__", silent = FALSE))

  item <- if (silent)
    tryCatch(getitem(key), error = function(e) NULL)
  else
    getitem(key)

  item
}

#' Set an item for a Python object
#'
#' Set an item on a Python object, similar to how
#' \code{x[name] = value} might be used in Python code to
#' set an item called `name` with value `value` on object
#' `x`. The object's `__setitem__` method will be called.
#'
#' @param x A Python object.
#' @param name The item name.
#' @param value The item value.
#'
#' @return The (mutated) object `x`, invisibly.
#'
#' @family item-related APIs
#' @export
py_set_item <- function(x, name, value) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)

  if (!py_has_attr(x, "__setitem__"))
    stop("Python object has no '__setitem__' method", call. = FALSE)
  setitem <- py_to_r(py_get_attr(x, "__setitem__", silent = FALSE))

  setitem(name, value)
  invisible(x)
}

#' Delete / remove an item from a Python object
#'
#' Delete an item associated with a Python object, as
#' through its `__delitem__` method.
#'
#' @param x A Python object.
#' @param name The item name.
#'
#' @return The (mutated) object `x`, invisibly.
#'
#' @family item-related APIs
#' @export
py_del_item <- function(x, name) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)

  if (!py_has_attr(x, "__delitem__"))
    stop("Python object has no '__delitem__' method", call. = FALSE)
  delitem <- py_to_r(py_get_attr(x, "__delitem__", silent = FALSE))

  delitem(name)
  invisible(x)
}



#' Unique identifer for Python object
#'
#' Get a globally unique identifer for a Python object.
#'
#' @note In the current implementation of CPython this is the
#'  memory address of the object.
#'
#' @param object Python object
#'
#' @return Unique identifer (as integer) or `NULL`
#'
#' @export
py_id <- function(object) {
  if (py_is_null_xptr(object) || !py_available())
    NULL
  else {
    py <- import_builtins()
    py$id(object)
  }
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

  # remove e.g. 'object at 0x10d084710'
  str <- gsub(" object at 0x\\w{4,}", "", str)

  # return
  str
}

#' @export
py_str.python.builtin.bytearray <- function(object, ...) {
  builtins <- import_builtins()
  paste0("python.builtin.bytearray (", builtins$len(object), " bytes)")
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
#' @param file Source file
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

  # collect module proxy hooks
  collect_value <- function(name) {
    if (exists(name, envir = proxy, inherits = FALSE)) {
      value <- get(name, envir = proxy, inherits = FALSE)
      remove(list = name, envir = proxy)
      value
    } else {
      NULL
    }
  }

  # name of module to import (allow just in time customization via hook)
  get_module <- collect_value("get_module")
  if (!is.null(get_module))
    assign("module", get_module(), envir = proxy)

  # get module name
  module <- get("module", envir = proxy)

  # load and error handlers
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

py_inject_r <- function(envir) {

  # define our 'R' class
  py_run_string("class R(object): pass")

  # extract it from the main module
  main <- import_main(convert = FALSE)
  R <- main$R

  # extract active knit environment
  if (is.null(envir)) {
    .knitEnv <- yoink("knitr", ".knitEnv")
    envir <- .knitEnv$knit_global
  }

  # define the getters, setters we'll attach to the Python class
  getter <- function(self, code) {
    object <- eval(parse(text = as_r_value(code)), envir = envir)
    r_to_py(object, convert = is.function(object))
  }

  setter <- function(self, name, value) {
    envir[[as_r_value(name)]] <<- as_r_value(value)
  }

  py_set_attr(R, "__getattr__", getter)
  py_set_attr(R, "__setattr__", setter)
  py_set_attr(R, "__getitem__", getter)
  py_set_attr(R, "__setitem__", setter)

  # now define the R object
  py_run_string("r = R()")

}

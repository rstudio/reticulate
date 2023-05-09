
#' @export
print.python.builtin.object <- function(x, ...) {
  writeLines(py_repr(x))
  invisible(x)
}


#' @importFrom utils str
#' @export
str.python.builtin.object <- function(object, ...) {
  if (!py_available() || py_is_null_xptr(object))
    cat("<pointer: 0x0>\n")
  else
    cat(py_str(object), "\n", sep = "")
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
#'  which raises an error. Other possible values are 'ignore' and 'replace'.
#'
#' @export
as.character.python.builtin.bytes <- function(x, encoding = "utf-8", errors = "strict", ...) {
  x$decode(encoding = encoding, errors = errors)
}


.operators <- new.env(parent = emptyenv())

fetch_op <- function(nm, .op, nargs = 1L) {
  ensure_python_initialized()
  if (is.null(fn <- .operators[[nm]])) {
    force(.op)

    if (is.function(.op))
      op <- .op
    else
      op <- function(...) py_call(.op, ...)

    if (nargs == 1L) {

      call_op_and_maybe_convert <- function(...)
        py_maybe_convert(op(...),  py_has_convert(..1))

    } else if (nargs == 2L) {

      # Ops group generics
      call_op_and_maybe_convert <- function(...) {
        result <- op(...)
        # if either dispatch object has convert=FALSE, don't convert
        convert <-
          !((inherits(..1, "python.builtin.object") && isFALSE(py_has_convert(..1))) ||
            (inherits(..2, "python.builtin.object") && isFALSE(py_has_convert(..2))))
        py_maybe_convert(result, convert)
      }

    } else stop("invalid nargs value: ", nargs)

    fn <- .operators[[nm]] <- call_op_and_maybe_convert
  }
  fn
}

#' @export
"==.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("eq", py_eval("lambda e1, e2: e1 == e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
"!=.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("ne", py_eval("lambda e1, e2: e1 != e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
"<.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("lt", py_eval("lambda e1, e2: e1 < e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
">.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("gt", py_eval("lambda e1, e2: e1 > e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
">=.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("ge", py_eval("lambda e1, e2: e1 >= e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
"<=.python.builtin.object" <- function(e1, e2) {
  op <- fetch_op("le", py_eval("lambda e1, e2: e1 <= e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

# This uses PyObject_RichCompareBool(), which expects only py bools.
# It will throw an exception on, e.g., with numpy arrays,
# even though numpy.ndarray defines an __eq__() method.
py_compare <- function(a, b, op) {
  ensure_python_initialized()
  py_validate_xptr(a)
  if (!inherits(b, "python.builtin.object"))
    b <- r_to_py(b)
  py_validate_xptr(b)
  py_compare_impl(a, b, op)
}


#' @export
`+.python.builtin.object` <- function(e1, e2) {
  if (missing(e2)) {
    op <- fetch_op("pos", py_eval("lambda e1: +e1", convert = FALSE))
    return(op(e1))
  }

  op <- fetch_op("add", py_eval("lambda e1, e2: e1 + e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}


#' @export
`-.python.builtin.object` <- function(e1, e2) {
  if (missing(e2)) {
    op <- fetch_op("neg", py_eval("lambda e1: -e1", convert = FALSE))
    return(op(e1))
  }
  op <- fetch_op("sub", py_eval("lambda e1, e2: e1 - e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}


#' @export
`*.python.builtin.object` <-function(e1, e2) {
  op <- fetch_op("*", py_eval("lambda e1, e2: e1 * e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`/.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("/", py_eval("lambda e1, e2: e1 / e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`%/%.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("//", py_eval("lambda e1, e2: e1 // e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`%%.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("%", py_eval("lambda e1, e2: e1 % e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`^.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("pow", import_builtins(FALSE)$pow,
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`&.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("&", py_eval("lambda e1, e2: e1 & e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`|.python.builtin.object` <- function(e1, e2) {
  op <- fetch_op("|", py_eval("lambda e1, e2: e1 | e2", convert = FALSE),
                 nargs = 2L)
  op(e1, e2)
}

#' @export
`!.python.builtin.object` <- function(e1) {
  op <- fetch_op("~", py_eval("lambda e1: ~ e1", convert = FALSE))
  op(e1)
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

  # if this is already an R object, nothing to do
  if (!inherits(x, "python.builtin.object"))
    return(x)

  # if it's neither convertable nor callable,
  # nothing to do
  convertable <- convert || py_is_callable(x)
  if (!convertable)
    return(x)

  # perform conversion
  # capture previous convert for attr
  attrib_convert <- py_has_convert(x)

  # temporarily change convert so we can call py_to_r and get S3 dispatch
  envir <- as.environment(x)
  assign("convert", convert, envir = envir)
  on.exit(assign("convert", attrib_convert, envir = envir), add = TRUE)

  # call py_to_r
  py_to_r(x)

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
  object <- NULL
  if (prefer_attr) {
    object <- py_get_attr(x, name)
  } else {

    # if we have an attribute, attempt to get the item
    # but allow for fallback to that attribute. note that
    # the logic here is fairly convoluted but is necessary
    # to maintain backwards compatibility with a number of
    # CRAN packages (hopefully we can simplify this in the
    # future)
    if (py_has_attr(x, name)) {

      # try to get item
      if (py_has_attr(x, "__getitem__"))
        object <- py_get_item(x, name, silent = TRUE)

      # fallback to attribute
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
    types <- py_suppress_warnings(py_get_attr_types(x, names))
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

  idx <- grepl(pattern, names)
  names <- names[idx]
  types <- types[idx]

  if (length(names) > 0) {
    # set types
    oidx <- order(names)
    names <- names[oidx]
    attr(names, "types") <- types[oidx]

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
#'   from Python to R. You can do manual conversion with the [py_to_r()]
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
#'   Python to R. You can do manual conversion with the [py_to_r()] function or
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
#' Get the length of a Python object. This is equivalent to calling
#' the Python builtin `len()` function on the object.
#'
#' Not all Python objects have a defined length. For objects without a defined
#' length, calling `py_len()` will throw an error. If you'd like to instead
#' infer a default length in such cases, you can set the `default` argument
#' to e.g. `1L`, to treat Python objects without a `__len__` method as having
#' length one.
#'
#' @param x A Python object.
#'
#' @param default The default length value to return, in the case that
#'   the associated Python object has no `__len__` method. When `NULL`
#'   (the default), an error is emitted instead.
#'
#' @return The length of the object, as a numeric value.
#'
#' @export
py_len <- function(x, default = NULL) {

  # return 0 if Python not yet available
  if (py_is_null_xptr(x) || !py_available())
    return(0L)

  # delegate to C++
  py_len_impl(x, default)
}

#' @export
length.python.builtin.list <- function(x) {
  py_list_length(x)
}

#' @export
length.python.builtin.object <- function(x) {

  # return 0 if Python not yet available
  if (py_is_null_xptr(x) || !py_available())
    return(0L)

  # otherwise, try to invoke the object's __len__ method
  n <- py_len_impl(x, NA_integer_)
  if (is.na(n))
    # if the object didn't have a __len__ method, or __len__ raised an
    # Exception, try instead to invoke its __bool__ method
    return(as.integer(py_bool_impl(x)))

  n
}


#' Python Truthiness
#'
#' Equivalent to `bool(x)` in Python, or `not not x`.
#'
#' If the Python object defines a `__bool__` method, then that is invoked.
#' Otherwise, if the object defines a `__len__` method, then `TRUE` is
#' returned if the length is nonzero. If neither `__len__` nor `__bool__`
#' are defined, then the Python object is considered `TRUE`.
#'
#' @param x, A python object.
#'
#' @return An R scalar logical: `TRUE` or `FALSE`. If `x` is a
#'   null pointer or Python is not initialized, `FALSE` is returned.
#' @export
py_bool <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    FALSE
  else
    py_bool_impl(x)
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
#'   string object is returned).
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

  # TODO: would like to use PyIter_Check() but that is only implemented
  # as a macro in Python 2.x and requires copying more headers
  iterable <- py_has_attr(it, "__next__") || py_has_attr(it, "next")
  if (!iterable)
    stop("object is not iterable", call. = FALSE)

  py_iter_next(it, completed)

}


#' @rdname iterate
#' @export
as_iterator <- function(x) {
  if (inherits(x, "python.builtin.iterator"))
    x
  else if (py_has_attr(x, "__iter__"))
    x$`__iter__`()
  else
    stop("iterator function called with non-iterator argument", call. = FALSE)
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
  dots <- split_named_unnamed(list(...))
  py_call_impl(x, dots$unnamed, dots$named)
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

#' The Python None object
#'
#' Get a reference to the Python `None` object.
#'
#' @export
py_none <- function() {
  ensure_python_initialized()
  py_none_impl()
}

#' Delete an attribute of a Python object
#'
#' @param x A Python object.
#' @param name The attribute name.
#'
#' @export
py_del_attr <- function(x, name) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_del_attr_impl(x, name)
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

py_get_attr_types <- function(x,
                              names,
                              resolve_properties = FALSE)
{
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)

  py_get_attr_types_impl(x, names, resolve_properties)
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

  # NOTE: for backwards compatibility, we make sure to return an R NULL on error
  if (silent) {
    tryCatch(py_get_item_impl(x, key, FALSE), error = function(e) NULL)
  } else {
    py_get_item_impl(x, key, FALSE)
  }

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
  py_set_item_impl(x, name, value)
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





#' String representation of a python object.
#'
#' This is equivalent to calling `str(object)` or `repr(object)` in Python.
#'
#' In Python, calling `print()` invokes the builtin `str()`, while auto-printing
#' an object at the REPL invokes the builtin `repr()`.
#'
#' In \R, the default print method for python objects invokes `py_repr()`, and
#' the default `format()` and `as.character()` methods invoke `py_str()`.
#'
#' For historical reasons, `py_str()` is also an \R S3 method that allows R
#' authors to customize the the string representation of a Python object from R.
#' New code is recommended to provide a `format()` and/or `print()` S3 R method
#' for python objects instead.
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
    "<not a python object>"
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
  py_str_impl(object)
}


#' @export
format.python.builtin.object <- function(x, ...) {

  if (py_is_null_xptr(x) || !py_available())
    return("<pointer: 0x0>")

  # get default rep, potentially user defined S3
  str <- py_str(x)

  # remove e.g. 'object at 0x10d084710'
  str <- gsub(" object at 0x\\w{4,}", "", str)

  # return
  str
}

#' @export
py_str.python.builtin.bytearray <- function(object, ...) {
  paste0("python.builtin.bytearray (", py_len_impl(object), " bytes)")
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
  len <- py_len_impl(object)
  if (len > 10)
    paste0(name, " (", len, " items)")
  else
    py_str.python.builtin.object(object)
}

.print.via.format <- function(x, ...) {
  writeLines(format(x, ...))
  invisible(x)
}

#' @export
print.python.builtin.bytearray <- .print.via.format
#' @export
print.python.builtin.tuple <- .print.via.format
#' @export
print.python.builtin.module <- .print.via.format
#' @export
print.python.builtin.list <- .print.via.format
#' @export
print.python.builtin.dict <- .print.via.format


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

  # scope output capture
  capture_stdout <- "stdout" %in% type
  capture_stderr <- "stderr" %in% type
  output_tools$start_capture(capture_stdout, capture_stderr)
  on.exit(output_tools$end_capture(capture_stdout, capture_stderr), add = TRUE)

  # evaluate the expression
  force(expr)

  # collect output
  output_tools$collect_output()

}

py_flush_output <- function() {

  if (!is_python3())
    return()

  sys <- import("sys", convert = TRUE)

  if (!is.null(sys$stdout) && is.function(sys$stdout$flush))
    sys$stdout$flush()

  if (!is.null(sys$stderr) && is.function(sys$stderr$flush))
    sys$stderr$flush()

}



#' Run Python code
#'
#' Execute code within the scope of the \code{__main__} Python module.
#'
#' @inheritParams import
#'
#' @param code The Python code to be executed.
#' @param file The Python script to be executed.
#' @param local Boolean; should Python objects be created as part of
#'   a local / private dictionary? If `FALSE`, objects will be created within
#'   the scope of the Python main module.
#' @param prepend_path Boolean; should the script directory be added to the
#'   Python module search path? The default, `TRUE`, matches the behavior of
#'   `python <path/to/script.py>` at the command line.
#'
#' @return A Python dictionary of objects. When `local` is `FALSE`, this
#'   dictionary captures the state of the Python main module after running
#'   the provided code. Otherwise, only the variables defined and used are
#'   captured.
#'
#' @name py_run
#'
#' @export
py_run_string <- function(code, local = FALSE, convert = TRUE) {
  ensure_python_initialized()
  on.exit(py_flush_output(), add = TRUE)
  invisible(py_run_string_impl(code, local, convert))
}

#' @rdname py_run
#' @export
py_run_file <- function(file, local = FALSE, convert = TRUE, prepend_path = TRUE) {
  ensure_python_initialized()

  file <- path.expand(file)
  if (prepend_path) {
    sys <- import("sys", convert = FALSE)
    sys$path$insert(0L, dirname(file))
    on.exit(sys$path$remove(dirname(file)), add = TRUE)
  }
  invisible(py_run_file_impl(file, local, convert))
}

#' Evaluate a Python Expression
#'
#' Evaluate a single Python expression, in a way analogous to the Python
#' `eval()` built-in function.
#'
#' @param code A single Python expression.
#' @param convert Boolean; automatically convert Python objects to R?
#'
#' @return The result produced by evaluating `code`, converted to an `R`
#'   object when `convert` is set to `TRUE`.
#'
#' @section Caveats:
#'
#' `py_eval()` only supports evaluation of 'simple' Python expressions.
#' Other expressions (e.g. assignments) will fail; e.g.
#'
#' ```
#' > py_eval("x = 1")
#' Error in py_eval_impl(code, convert) :
#'   SyntaxError: invalid syntax (reticulate_eval, line 1)
#' ```
#'
#' and this mirrors what one would see in a regular Python interpreter:
#'
#' ```
#' >>> eval("x = 1")
#' Traceback (most recent call last):
#'   File "<stdin>", line 1, in <module>
#'   File "<string>", line 1
#' x = 1
#' ^
#'   SyntaxError: invalid syntax
#' ```
#'
#' The [py_run_string()] method can be used if the evaluation of arbitrary
#' Python code is required.
#'
#' @export
py_eval <- function(code, convert = TRUE) {
  ensure_python_initialized()
  py_eval_impl(code, convert)
}

#' The builtin constant Ellipsis
#'
#' @export
py_ellipsis <- function() {
  builtins <- import_builtins(convert = FALSE)
  builtins$Ellipsis
}

#' @importFrom rlang list2
py_callable_as_function <- function(callable, convert) {

  force(callable)
  force(convert)

  as.function.default(c(py_get_formals(callable), quote({
    cl <- sys.call()
    cl[[1L]] <- list2

    call_args <- split_named_unnamed(eval(cl, parent.frame()))
    result <- py_call_impl(callable, call_args$unnamed, call_args$named)

    if (convert)
      result <- py_to_r(result)

    if (is.null(result))
      invisible(result)
    else
      result
  })))
}


split_named_unnamed <- function(x) {
  nms <- names(x)
  if (is.null(nms))
    return(list(unnamed = x, named = list()))
  named <- nzchar(nms)
  list(unnamed = x[!named], named = x[named])
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
  before_load <- collect_value("before_load")
  on_load <- collect_value("on_load")
  on_error <- collect_value("on_error")

  # execute before load handler
  if (is.function(before_load))
    before_load()

  # perform the import -- capture error and amend it with
  # python configuration information if we have it
  result <- tryCatch(import(module), error = clear_error_handler())
  if (inherits(result, "error")) {
    if (!is.null(on_error)) {

      # call custom error handler
      if (is.function(on_error))
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

  # call on_load if provided
  if (is.function(on_load))
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

py_inject_r <- function() {

  # don't inject 'r' if there's already an 'r' object defined
  main <- import_main(convert = FALSE)
  if (py_has_attr(main, "r"))
    return(FALSE)

  # define our 'R' class
  py_run_string("class R(object): pass")

  # extract it from the main module
  main <- import_main(convert = FALSE)
  R <- main$R

  # define the getters, setters we'll attach to the Python class
  getter <- function(self, code) {
    envir <- py_resolve_envir()
    object <- eval(parse(text = as_r_value(code)), envir = envir)
    r_to_py(object, convert = is.function(object))
  }

  setter <- function(self, name, value) {
    envir <- py_resolve_envir()
    name  <- as_r_value(name)
    value <- as_r_value(value)
    assign(name, value, envir = envir)
  }

  py_set_attr(R, "__getattr__", getter)
  py_set_attr(R, "__setattr__", setter)
  py_set_attr(R, "__getitem__", getter)
  py_set_attr(R, "__setitem__", setter)

  # now define the R object
  py_run_string("r = R()")

  # remove the 'R' class object
  py_del_attr(main, "R")

  # indicate success
  TRUE

}

py_resolve_envir <- function() {

  # if an environment has been set, use it
  envir <- getOption("reticulate.engine.environment")
  if (is.environment(envir))
    return(envir)

  # if we're running in a knitr document, use the knit env
  if ("knitr" %in% loadedNamespaces()) {
    .knitEnv <- yoink("knitr", ".knitEnv")
    envir <- .knitEnv$knit_global
    if (is.environment(envir))
      return(envir)
  }

  # if we're running in a testthat test, use the rlang reported envir
  envir <- getOption("rlang_trace_top_env")
  if (is.environment(envir))
    return(envir)

  # otherwise, default to the global environment
  envir %||% globalenv()

}

py_inject_hooks <- function() {

  builtins <- import_builtins(convert = TRUE)

  input <- function(prompt = "") {

    response <- tryCatch(
      readline(prompt),
      interrupt = identity
    )

    if (inherits(response, "interrupt"))
      stop("KeyboardInterrupt", call. = FALSE)

    r_to_py(response)

  }

  # override input function
  if (interactive() && was_python_initialized_by_reticulate()) {
    name <- if (is_python3()) "input" else "raw_input"
    builtins[[name]] <- input
  }

  # register module import callback
  useImportHook <- getOption("reticulate.useImportHook", default = is_python3())
  if (useImportHook) {
    loader <- import("rpytools.loader", convert = TRUE)
    loader$initialize(py_module_onload)
  }

}

py_module_onload <- function(module) {

  # log module loading if requested
  if (getOption("reticulate.logModuleLoad", default = FALSE)) {
    writeLines(sprintf("Loaded module '%s'", module))
  }

  # retrieve and clear list of hooks
  hookName <- paste("reticulate", module, "load", sep = "::")
  hooks <- getHook(hookName)
  setHook(hookName, NULL, action = "replace")

  # run hooks
  for (hook in hooks)
    tryCatch(hook(), error = warning)

}

py_module_loaded <- function(module) {
  sys <- import("sys", convert = TRUE)
  modules <- sys$modules
  module %in% names(modules)
}

py_register_load_hook <- function(module, hook) {

  # if the module is already loaded, just run the hook
  if (py_module_loaded(module))
    return(hook())

  # otherwise, register the hook to be run on next load
  name <- paste("reticulate", module, "load", sep = "::")
  setHook(name, hook)

}

py_set_interrupt <- function() {
  py_set_interrupt_impl()
}

#' @export
format.python.builtin.traceback <- function(x, ..., limit = NULL) {
  import("traceback")$format_tb(x, limit)
}


#' @rdname py_last_error
#' @export
py_clear_last_error <- function() {
  py_last_error(NULL)
}

#' Get or (re)set the last Python error encountered.
#'
#' @param exception A python exception object. If provided, the provided
#'   exception is set as the last exception.
#'
#' @return For `py_last_error()`, `NULL` if no error has yet been encountered.
#'   Otherwise, a named list with entries:
#'
#'   +  `"type"`: R string, name of the exception class.
#'
#'   +  `"value"`: R string, formatted exception message.
#'
#'   +  `"traceback"`: R character vector, the formatted python traceback,
#'
#'   +  `"message"`: The full formatted raised exception, as it would be printed in
#'   Python. Includes the traceback, type, and value.
#'
#' And attribute `"exception"`, a `'python.builtin.Exception'` object.
#'
#' The named list has `class` `"py_error"`, and has a default `print` method
#' that is the equivalent of `cat(py_last_error()$message)`.
#'
#' @examples
#' \dontrun{
#' # run python code that might error,
#' # without modifying the user-visible python exception
#'
#' safe_len <- function(x) {
#'   last_err <- py_last_error()
#'   tryCatch({
#'     # this might raise a python exception if x has no `__len__` method.
#'     import_builtins()$len(x)
#'   }, error = function(e) {
#'     # py_last_error() was overwritten, is now "no len method for 'object'"
#'     py_last_error(last_err) # restore previous exception
#'     -1L
#'   })
#' }
#'
#' safe_len(py_eval("object"))
#' }
#'
#' @export
py_last_error <- function(exception) {
  if (!missing(exception)) {
    # set as the last exception
    r_trace <- NULL
    if (inherits(exception, "py_error")) {
      r_trace <- exception$r_trace
      exception <- attr(exception, "exception", TRUE)
    }

    if(is.null(r_trace))
      r_trace <- as_r_value(py_get_attr(exception, "r_trace", TRUE))

    if (!is.null(exception) &&
        !inherits(exception, "python.builtin.Exception"))
      stop("`exception` must be NULL, a `py_error`, or a 'python.builtin.Exception'")

    on.exit({
      .globals$py_last_exception <- exception
      .globals$last_r_trace <- r_trace
      })
    return(invisible(.globals$py_last_exception))
  }

  e <- .globals$py_last_exception

  if (is.null(e))
    return(NULL)

  if (!py_available() || py_is_null_xptr(e)) {
    .globals$py_last_exception <- NULL
    return(NULL)
  }

  etype <- py_get_attr_impl(e, "__class__")
  etb <- py_get_attr_impl(e, "__traceback__", TRUE)
  traceback <- import("traceback")

  if(is.null(etb))
    formatted_traceback <- NULL
  else
    formatted_traceback <- traceback$format_tb(etb)

  out <- list(
    type = py_get_attr_impl(etype, "__name__", TRUE),
    value = py_str_impl(e),
    traceback = formatted_traceback,
    message = paste0(traceback$format_exception(etype, e, etb),
                     collapse = "")
  )
  out$r_call <- conditionCall(e)
  out$r_class <- as_r_value(py_get_attr(e, "r_class", TRUE)) %||% class(e)
  out$r_trace <- py_get_attr(e, "r_trace", TRUE) %||% .globals$last_r_trace
  out <- lapply(out, as_r_value)
  attr(out, "exception") <- e
  class(out) <- "py_error"
  out
}



make_filepaths_clickable <- function(formatted_python_traceback) {
  # Note, a first draft of this iterated over the list of FrameSummarys in
  # the exception.__traceback__, but that approach breaks with keras.
  # So now we use a regex instead (:sad:).
  # See format_py_exception_traceback_with_clickable_filepaths()
  # for the previous approach

  x <- strsplit(formatted_python_traceback, "\n", fixed = TRUE)[[1L]]
  if (!length(x))
    return(formatted_python_traceback)
  m <- regexec('File "([^"]+)", line ([0-9]+), in', x, perl = TRUE)

  new <- lapply(regmatches(x, m), function(match) {
    if (!length(match))
      return(character())
    filepath <- match[2]
    lineno <- match[3]
    link <- cli::style_hyperlink(
      filepath,
      paste0("file://", normalizePath(filepath, mustWork = FALSE)),
      params = c(line = lineno))
    cli::col_grey(link)
  })

  m2 <- lapply(m, function(match_pos) {
    if(identical(as.vector(match_pos), -1L))
      return(match_pos)
    out <- match_pos[2] # only match filepath
    # TODO, make the clickable target bigger, include ", line nn" in link.
    attr(out, "match.length") <- attr(match_pos, "match.length")[2]
    out
  })

  regmatches(x, m2) <- new

  if(x[length(x)] != "")
    x <- c(x, "") # ensure we end w/ a newline
  paste0(x, collapse = "\n")
}

#' @export
print.py_error <- function(x, ...) {

  py_error_message <- x$message

  if (identical(.Platform$GUI, "RStudio") &&
      requireNamespace("cli", quietly = TRUE) &&
      length(etb <- attr(x, "exception")$`__traceback__`))
    py_error_message <- make_filepaths_clickable(py_error_message)

  cat_h1("Python Exception Message")
  cat(py_error_message)

  cat_h1("R Traceback")
  print(x$r_trace)
}

cat_h1 <- function(x) {
  if(requireNamespace("cli", quietly = TRUE)) {
    cli::cli_h1(x, .envir = NULL)
  } else {
    cat("--- ", x, "\n", sep = "")
  }
}

format_py_exception_traceback_with_clickable_filepaths <- function(etb) {
  # This is currently unused, but preserved here in case it's useful for future
  # development. This is unused because keras/tensorflow hijacks the python
  # exception __traceback__, making it effectively useless. Instead, keras
  # formats the actual (user relevant) traceback info directly into the
  # exception message (and nicely too! albeit verbosely. It includes detailed
  # info about call args in each user frame, including tensor shapes and dtypes,
  # and formats with indentation matching user-generated frame depth).
  # Unfortunately, that means that building up a nice formatted traceback by
  # iterating over the traceback FrameSummary objects won't work correctly. The
  # alternative is to apply a regex to the message, as we do in
  # make_filepaths_clickable() (:sad:)

  if(is.null(etb)) return(NULL)
  fsl <- import("traceback")$extract_tb(etb)
  if(!length(fsl)) return(NULL)
  paste0(collapse = "\n", c(
    "Traceback (most recent call last):",
    vapply(fsl, function(fs) {
      # fs == FrameSummary obj, with attrs: filename, line, lineno, locals, name
      filepath <- fs$filename
      lineno <- fs$lineno
      clickable_filepath <-
        cli::style_hyperlink(
          filepath,
          paste0("file://", normalizePath(filepath, mustWork = FALSE)),
          params = c(line = lineno)
        )
      sprintf('  File "%s", line %i, in %s\n    %s',
              clickable_filepath, lineno, fs$name, fs$line)
    }, ""),
    ""))
}


.py_last_error_hint <- function() {

  if(!interactive() ||
     !identical(.Platform$GUI, "RStudio") ||
     !requireNamespace("cli", quietly = TRUE))
    return("Run `reticulate::py_last_error()` for details.")

  py_last_error <- cli::style_hyperlink(
    "`reticulate::py_last_error()`",
    "rstudio:run:reticulate::py_last_error()")

  cli::col_silver(paste("Run", py_last_error, "for details."))
}

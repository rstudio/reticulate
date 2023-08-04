


#' Get/Set/Delete an item from a Python object
#'
#' Access an item from a Python object, similar to how \code{x[key]} might be
#' used in Python code to access an item indexed by `key` on an object `x`. The
#' object's `__getitem__()` `__setitem__()` or `__delitem__()` method will be
#' called.
#'
#' @note The `py_get_item()` always returns an unconverted python object, while
#'   `[` will automatically attempt to convert the object if `x` was created
#'   with `convert = TRUE`.
#'
#' @param x A Python object.
#' @param key,name,... The key used for item lookup.
#' @param silent Boolean; when \code{TRUE}, attempts to access missing items
#'   will return \code{NULL} rather than throw an error.
#' @param value The item value to set. Assigning `value` of `NULL` calls
#'   `py_del_item()` and is equivalent to the python expression `del x[key]`. To
#'   set an item value of `None`, you can call `py_set_item()` directly, or call
#'   `x[key] <- py_none()`
#'
#' @return For `py_get_item()` and `[`, the return value from the
#'   `x.__getitem__()` method. For `py_set_item()`, `py_del_item()` and `[<-`,
#'   the mutate object `x` is returned.
#'
#' @rdname py_get_item
#' @family item-related APIs
#' @export
#' @examples
#' \dontrun{
#'
#' ## get/set/del item from Python dict
#' x <- r_to_py(list(abc = "xyz"))
#'
#' #'   # R expression    | Python expression
#' # -------------------- | -----------------
#'  x["abc"]              # x["abc"]
#'  x["abc"] <- "123"     # x["abc"] = "123"
#'  x["abc"] <- NULL      # del x["abc"]
#'  x["abc"] <- py_none() # x["abc"] = None
#'
#' ## get item from Python list
#' x <- r_to_py(list("a", "b", "c"))
#' x[0]
#'
#' ## slice a NumPy array
#' x <- np_array(array(1:64, c(4, 4, 4)))
#'
#' # R expression | Python expression
#' # ------------ | -----------------
#'   x[0]         # x[0]
#'   x[, 0]       # x[:, 0]
#'   x[, , 0]     # x[:, :, 0]
#'
#'   x[NA:2]      # x[:2]
#'   x[`:2`]      # x[:2]
#'
#'   x[2:NA]      # x[2:]
#'   x[`2:`]      # x[2:]
#'
#'   x[NA:NA:2]   # x[::2]
#'   x[`::2`]     # x[::2]
#'
#'   x[1:3:2]     # x[1:3:2]
#'   x[`1:3:2`]   # x[1:3:2]
#'
#' }
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

#' @rdname py_get_item
#' @export
py_set_item <- function(x, name, value) {
  ensure_python_initialized()
  if (py_is_module_proxy(x))
    py_resolve_module_proxy(x)
  py_set_item_impl(x, name, value)
  invisible(x)
}

#' @rdname py_get_item
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


#' @rdname py_get_item
#' @export
`[.python.builtin.object` <- function(x, ...) {

  key <- dots_to__getitem__key(..., .envir = parent.frame())

  out <- if(inherits(key, "python.builtin.tuple"))
    py_get_item(x, key)
  else
    py_get_attr_or_item(x, key, FALSE) # prefer_attr = FALSE
  py_maybe_convert(out, py_has_convert(x))
}

#' @rdname py_get_item
#' @export
`[<-.python.builtin.object` <- function(x, ..., value) {
  if (py_is_null_xptr(x) || !py_available())
    stopf("Unable to assign value (`%s` reference is NULL)", deparse1(substitute(x)))

  key <- dots_to__getitem__key(..., .envir = parent.frame())

  if(is.null(value))
    py_del_item(x, key)
  else
    py_set_item(x, key, value)
}


dots_to__getitem__key <- function(..., .envir) {
  dots <- lapply(eval(substitute(alist(...))), function(d) {

    if(is_missing(d))
      return(py_slice())

    if (is_has_colon(d)) {

      if (is_colon_call(d)) {

        d <- as.list(d)[-1L]

        if (is_colon_call(d[[1L]] -> d1)) # step supplied
          d <- c(as.list(d1)[-1L], d[-1L])

      } else { # single name with colon , like `::2`

        d <- deparse(d, width.cutoff = 500L, backtick = FALSE)
        d <- strsplit(d, ":", fixed = TRUE)[[1L]]
        d[!nzchar(d)] <- "NULL"
        d <- lapply(d, parse1) # rlang::parse_expr
      }

      if(!length(d) %in% 1:3)
        stop("Only 1, 2, or 3 arguments can be supplied as a python slice")

      d <- lapply(d, eval, envir = .envir)
      d <- lapply(d, function(e) if(identical(e, NA) ||
                                    identical(e, NA_integer_) ||
                                    identical(e, NA_real_)) NULL else e)

      return(do.call(py_slice, d))
    }

    # else, eval normally
    d <- eval(d, envir = .envir)
    if(rlang::is_scalar_integerish(d))
      d <- as.integer(d)
    d
  })

  if(length(dots) == 1L)
    dots[[1L]]
  else
    tuple(dots)
}

# TODO: update these to use rlang
is_has_colon <- function(x)
  is_colon_call(x) || (is.symbol(x) && grepl(":", as.character(x), fixed = TRUE))

is_colon_call <- function(x)
  is.call(x) && identical(x[[1L]], quote(`:`))

is_missing <- function(x) identical(x, quote(expr =))

parse1 <- function (text)  parse(text = text, keep.source = FALSE)[[1L]]


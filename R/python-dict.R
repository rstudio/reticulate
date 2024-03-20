#' @export
`$.python.builtin.dict` <- function(x, name) {
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)

  if (py_has_attr(x, name)) {
    item <- py_get_attr(x, name)
    return(py_maybe_convert(item, py_has_convert(x)))
  }

  `[.python.builtin.dict`(x, name)
}

#' @export
`[.python.builtin.dict` <- function(x, name) {
  if (py_is_null_xptr(x) || !py_available())
    return(NULL)

  py_dict_get_item(x, name)
}

#' @export
`[[.python.builtin.dict` <- `[.python.builtin.dict`

#' @export
`$<-.python.builtin.dict` <- function(x, key, value) {
  if (py_is_null_xptr(x) || !py_available())
    stop("Unable to assign value (dict reference is NULL)")

  if(is.null(value))
    py_del_item(x, key)
  else
    py_dict_set_item(x, key, value)
  x
}

#' @export
`[<-.python.builtin.dict` <- `$<-.python.builtin.dict`

#' @export
`[[<-.python.builtin.dict` <- `$<-.python.builtin.dict`

#' @export
length.python.builtin.dict <- function(x) {
  if (py_is_null_xptr(x) || !py_available())
    0L
  else
    py_dict_length(x)
}

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

  item <- py_dict_get_item(x, name)
  py_maybe_convert(item, py_has_convert(x))
}

#' @export
`[[.python.builtin.dict` <- `[.python.builtin.dict`

#' @export
`$<-.python.builtin.dict` <- function(x, name, value) {
  if (!py_is_null_xptr(x) && py_available())
    py_dict_set_item(x, name, value)
  else
    stop("Unable to assign value (dict reference is NULL)")
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

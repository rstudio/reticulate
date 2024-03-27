#' @export
`$.python.builtin.dict` <- function(x, name) {
  attr <- py_get_attr(x, name, TRUE)
  if(is.null(attr))
    py_dict_get_item(x, name)
  else
    py_maybe_convert(attr, py_has_convert(x))
}

#' @export
`[.python.builtin.dict` <- function(x, name) {
  py_dict_get_item(x, name)
}

#' @export
`[[.python.builtin.dict` <- `[.python.builtin.dict`

#' @export
`$<-.python.builtin.dict` <- function(x, key, value) {
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

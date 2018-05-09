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

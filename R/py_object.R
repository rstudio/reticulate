
#' @export
`$.py_object` <- function(x, name) {
  attr <- py_object_get_attr(x, name)
  if (py_object_is_callable(attr)) {
    function(...) {
      py_object_call(attr)
    }
  } else {
    attr
  }
}

# Alias to [[
#' @export
`[[.py_object` <- `$.py_object`

# Completion
#' @export
.DollarNames.py_object <- function(x, pattern = "") {
  attrs <- py_list_attributes(x)
  attrs[substr(attrs, 1, 1) != '_']
}

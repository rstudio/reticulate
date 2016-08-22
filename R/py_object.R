
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


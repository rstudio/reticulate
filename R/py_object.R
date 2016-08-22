
#' @export
`$.py_object` <- function(x, name) {

  # if the name exists then return it
  if (name %in% ls(x, all.names = TRUE, sorted = FALSE)) {
    .subset2(x, name)

    # otherwise dynamically dispatch to the object instance
  } else {

    attr <- py_object_get_attr(x, name)
    if (py_object_is_callable(attr)) {
      function(...) {
        py_object_call(attr)
      }
    } else {
      attr
    }
  }
}


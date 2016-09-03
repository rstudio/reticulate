#' Create a constant tensor.
#'
#' The resulting tensor is populated with values of type \code{dtype}, as
#' specified by arguments \code{value} and (optionally) \code{shape} (see
#' examples below).
#'
#' @param value A constant value (or list) of output type \code{dtype}.
#' @param dtype The type of the elements of the resulting tensor.
#' @param shape Optional dimensions of resulting tensor.
#' @param name Optional name for the tensor.
#'
#' @return A Constant Tensor.
#'
#' @details
#' The argument \code{value} can be a constant value, or a list of values of
#' type \code{dtype}. If \code{value} is a list, then the length of the list
#' must be less than or equal to the number of elements implied by the
#' \code{shape} argument (if specified). In the case where the list length is
#' less than the number of elements specified by \code{shape}, the last element
#' in the list will be used to fill the remaining entries.
#'
#' The argument \code{shape} is optional. If present, it specifies the
#' dimensions of the resulting tensor. If not present, the shape of \code{value}
#' is used.
#'
#' If the argument \code{dtype} is not specified, then the type is inferred from
#' the type of \code{value}.
#'
#' @export
constant <- function(value, dtype=NULL, shape=NULL, name="Const") {
  tf$constant(value, dtype = as_dtype(dtype), shape = shape, name = name)
}

#' @export
variable <- function(initial_value=NULL, trainable=TRUE, collections=NULL,
                     validate_shape=TRUE, caching_device=NULL, name=NULL,
                     variable_def=NULL, dtype=NULL) {
  tf$Variable(initial_value,
              trainable = trainable,
              collections = collections,
              validate_shape = validate_shape,
              caching_device = caching_device,
              name = name,
              variable_def = variable_def,
              dtype = as_dtype(dtype))
}

#' @export
zeros <- function(shape, dtype="float32", name=NULL) {
  tf$zeros(as_shape(shape), dtype = as_dtype(dtype), name = name)
}

#' @export
print.tensorflow.python.framework.ops.Tensor <- function(x, ...) {
  py_print(x)
  if (!is.null(tf$get_default_session())) {
    cat("\n")
    cat(py_print(x$eval()))
  }
}

as_dtype <- function(dtype) {
  if (!is.null(dtype))
    tf$as_dtype(dtype)
  else
    NULL
}

as_integer <- function(x) {
  if (is.null(x))
    NULL
  else
    as.integer(x)
}

as_list <- function(x) {
  if (is.null(x))
    NULL
  else if (length(x) == 1)
    list(x)
  else
    x
}

as_shape <- function(shape) {
  as_list(as_integer(shape))
}


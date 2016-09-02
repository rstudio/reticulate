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
  tf <- tf_import()
  if (!is.null(dtype))
    dtype <- tf$as_dtype(dtype)
  tf$constant(value, dtype = dtype, shape = shape, name = name)
}

#' @export
print.tensorflow.python.framework.ops.Tensor <- function(x, ...) {
  py_print(x, ...)
  tf <- tf_import()
  if (!is.null(tf$get_default_session())) {
    cat("\n")
    cat(py_print(x$eval()))
  }
}

# Some math generics, see here for docs on more generics:
# https://stat.ethz.ch/R-manual/R-devel/library/base/html/groupGeneric.html

#' @export
"+.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$add(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"*.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$mul(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$mod(a, b)
}



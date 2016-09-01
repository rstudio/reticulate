
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





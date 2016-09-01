

#' @export
"+.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf <- tf_import()
  tf$add(a, b)
}



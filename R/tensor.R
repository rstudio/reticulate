
#' @export
tf.Variable <- function(initial_value=NULL, trainable=TRUE, collections=NULL,
                        validate_shape=TRUE, caching_device=NULL, name=NULL,
                        variable_def=NULL, dtype=NULL) {
  tf$Variable(initial_value,
              trainable = trainable,
              collections = collections,
              validate_shape = validate_shape,
              caching_device = caching_device,
              name = name,
              variable_def = variable_def,
              dtype = dtype)
}

#' @export
tf.constant <- function(value, dtype=NULL, shape=NULL, name="Const") {
  tf$constant(value, dtype = dtype, shape = shape, name = name)
}

#' @export
tf.placeholder <- function(dtype=NULL, shape=NULL, name=NULL) {
  tf$placeholder(dtype = dtype, shape = shape, name = name)
}

#' @export
tf.cast <- function(x, dtype, name=NULL) {
  tf$cast(x, dtype = dtype, name = name)
}

#' @export
tf.zeros <- function(shape, dtype=tf.float32, name=NULL) {
  tf$zeros(shape, dtype = dtype, name = name)
}

#' @export
print.tensorflow.python.framework.ops.Tensor <- function(x, ...) {
  py_print(x)
  if (!is.null(tf$get_default_session())) {
    cat("\n")
    cat(py_print(x$eval()))
  }
}



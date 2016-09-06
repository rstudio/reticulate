
# TODO: Share S3 methods for Variable and Tensor (and Placeholder?)
#        - may want to do a generic mapper and require that for
#          any extra classes to be added

# TODO: Documentation for existing methods

# TODO: API issues:
#    - use train_ prefix (where to draw the line?)
#    - perhaps use dots to namespace, e.g. tf.session?
#    -    tf.Session, tf.session
#    - consider wrapping objects as S4?
#    - Explosion of namespaces with nn, slim, tflearn, etc.
#      means it will be hard to not reflect them in our names
#      identical names may be best! (allows easy translation)

#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

# package level tf instance
tf <- NULL

# package level globals
.globals <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # load python
  py_initialize(pythonSharedLibrary());

  # import tensorflow api
  tf <<- py_import("tensorflow")

  # alias data types
  tf.float16 <<- tf$float16
  tf.float16_ref <<- tf$float16_ref
  tf.float32 <<- tf$float32
  tf.float32_ref <<- tf$float32_ref
  tf.float64 <<- tf$float64
  tf.float64_ref <<- tf$float64_ref
  tf.bfloat16 <<- tf$bfloat16
  tf.bfloat16_ref <<- tf$bfloat16_ref
  tf.complex64 <<- tf$complex64
  tf.complex64_ref <<- tf$complex64_ref
  tf.complex128 <<- tf$complex128
  tf.complex128_ref <<- tf$complex128_ref
  tf.int8 <<- tf$int8
  tf.int8_ref <<- tf$int8_ref
  tf.uint8 <<- tf$uint8
  tf.uint8_ref <<- tf$uint8_ref
  tf.int16 <<- tf$int16
  tf.int16_ref <<- tf$int16_ref
  tf.uint16 <<- tf$uint16
  tf.uint16_ref <<- tf$uint16_ref
  tf.int32 <<- tf$int32
  tf.int32_ref <<- tf$int32_ref
  tf.int64 <<- tf$int64
  tf.int64_ref <<- tf$int64_ref
  tf.bool <<- tf$bool
  tf.bool_ref <<- tf$bool_ref
  tf.qint8 <<- tf$qint8
  tf.qint8_ref <<- tf$qint8_ref
  tf.quint8 <<- tf$quint8
  tf.quint8_ref <<- tf$quint8_ref
  tf.qint16 <<- tf$qint16
  tf.qint16_ref <<- tf$qint16_ref
  tf.quint16 <<- tf$quint16
  tf.quint16_ref <<- tf$quint16_ref
  tf.qint32 <<- tf$qint32
  tf.qint32_ref <<- tf$qint32_ref
}

.onUnload <- function(libpath) {
  py_finalize();
}




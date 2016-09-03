
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
  py_initialize();
  tf <<- py_import("tensorflow")
}

.onUnload <- function(libpath) {
  py_finalize();
}




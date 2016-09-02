
# TODO: Share S3 methods for Variable and Tensor (and Placeholder?)
#        - may want to do a generic mapper and require that for
#          any extra classes to be added

# TODO: Documentation for existing methods

# TODO: API issues:
#    - use train_ prefix (where to draw the line?)

#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL


.onLoad <- function(libname, pkgname) {
  py_initialize();
}

.onUnload <- function(libpath) {
  py_finalize();
}

# package level globals
.globals <- new.env(parent = emptyenv())

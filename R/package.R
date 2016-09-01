
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

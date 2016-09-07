
#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {
  py_initialize(pythonSharedLibrary());
}

.onUnload <- function(libpath) {
  py_finalize();
}


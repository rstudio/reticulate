
#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {
  py_initialize(pythonSharedLibrary());
}

.onUnload <- function(libpath) {
  py_finalize();
}

# convenience function for importing tensorflow
#' @export
tensorflow <- function(module = NULL) {
  if (is.null(module))
    module <- "tensorflow"
  else
    module <- paste("tensorflow", module, sep=".")
  py_import(module)
}




# TODO: helper function for reading mnist data into named list
#
# TODO: 'train_accuracy' in minst_tutorial_pros is a numpy scalar, this
#       needs to become an R vector of the appropriate type (currently
#       it remains unconverted)

#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {

  # initialize python
  config <- py_config()
  py_initialize(config$libpython);

  # add our python scripts to the search path
  py_run_string(paste0("import sys; sys.path.append('",
                       system.file("python", package = "tensorflow") ,
                       "')"))

  # call tf onLoad handler
  tf_on_load(libname, pkgname)
}


.onAttach <- function(libname, pkgname) {
  # call tf onAttach handler
  tf_on_attach(libname, pkgname)
}

.onUnload <- function(libpath) {
  py_finalize();
}

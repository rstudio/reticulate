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

  # attempt to load tensorflow
  tf <<- import("tensorflow", silent = TRUE)

  # if we loaded tensorflow then register tf help topics
  if (!is.null(tf))
    register_tf_help_topics()
}


.onAttach <- function(libname, pkgname) {
  if (is.null(tf)) {
    packageStartupMessage("TensorFlow not currently installed, please see ",
                          "https://www.tensorflow.org/get_started/")
  }
}

.onUnload <- function(libpath) {
  py_finalize();
}

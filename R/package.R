
# TODO: add TENSORFLOW_PYTHON environment variable

# TODO: ? operator for help

# TODO: tensorflow.object or tensorflow.builtin.object rather than
#       tensorflow.python.object

# TODO: completion for np$absolute causes an error

# TODO: add docs on TENSORFLOW_PYTHON_VERSION python3
# TODO: revise modules section in api doc
# TODO: port selected "how to" docs
# TODO: port additional tutorials


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

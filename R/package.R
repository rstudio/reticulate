
# TODO: completion for Namespace objects (FLAGS)

# TODO: FLAGS <- parse_arguments(
#   argument("--foo", ...),
#   argument("--bar", ...),
#   foo = c(type = "character", ...),
#   bar = c(type = "integer", ...)
# )
# Implementation:
#   - Create argument parser and call add_argument
#   - Execute equivalent of tf.app.run()
#   - Could the function just be named "flags"? May be better
#     since the tf.app.run side effect will also occur and
#     that's not an intuitive outcome from "parse_arguments"
#
#

#' TensorFlow for R
#'
#' \href{https://tensorflow.org}{TensorFlow} is an open source software library
#' for numerical computation using data flow graphs. Nodes in the graph
#' represent mathematical operations, while the graph edges represent the
#' multidimensional data arrays (tensors) communicated between them. The
#' flexible architecture allows you to deploy computation to one or more CPUs or
#' GPUs in a desktop, server, or mobile device with a single API.
#'
#' The \href{https://www.tensorflow.org/api_docs/python/index.html}{TensorFlow
#' API} is composed of a set of Python modules that enable constructing and
#' executing TensorFlow graphs. The tensorflow package provides access to the
#' complete TensorFlow API from within R.
#'
#' For additional documentation on the tensorflow package see
#' \href{https://rstudio.github.io/tensorflow}{https://rstudio.github.io/tensorflow}
#'
#'
#' @docType package
#' @name tensorflow
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

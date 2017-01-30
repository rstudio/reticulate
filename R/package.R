
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

# record of tf config and load error message
.tf_config <- NULL
.py_bindings <- FALSE
.load_error_message <- NULL

.onLoad <- function(libname, pkgname) {

  # find configuration
  config <- tf_discover_config()
  .tf_config <<- config

  # check for basic python prerequsities
  if (is.null(config)) {
    .load_error_message <<- "Installation of Python not found, Python bindings not loaded."
    return()
  } else if (!file.exists(config$libpython)) {
    .load_error_message <<- paste0("Python shared library '", config$libpython, "' not found, Python bindings not loaded.")
    return()
  } else if (is.null(config$numpy) || config$numpy$version < "1.11") {
    .load_error_message <<- "Installation of Numpy >= 1.11 not found, Python bindings not loaded."
    return()
  } else if (config$anaconda) {
    .load_error_message <<- paste0("The tensorflow package does not support Anaconda distributions of Python.\n",
                                   "Please install tensorflow within another version of Python.")
    return()
  } else if (!has_compatible_arch(config)) {
    .load_error_message <<- paste0("Your current architecture is ", python_arch(), " however this version of ",
                                   "Python is compiled for ", config$architecture, ".")
    return()
  }

  # initialize python
  py_initialize(config$python,
                config$libpython,
                config$pythonhome,
                config$version >= "3.0");

  # set internal flag indicating we have py bindings
  .py_bindings <<- TRUE

  # add our python scripts to the search path
  py_run_string(paste0("import sys; sys.path.append('",
                       system.file("python", package = "tensorflow") ,
                       "')"))

  # attempt to load tensorflow
  tf <<- tryCatch(import("tensorflow"), error = function(e) e)
  if (inherits(tf, "error")) {
    .load_error_message <<- tf$message
    tf <<- NULL
  }

  # if we loaded tensorflow then register tf help topics
  if (!is.null(tf))
    register_tf_help_topics()
}


.onAttach <- function(libname, pkgname) {

  if (is.null(tf)) {
    packageStartupMessage("\n", .load_error_message)
    packageStartupMessage("\nIf you have not yet installed TensorFlow, see ",
                          "https://www.tensorflow.org/get_started/\n")
    packageStartupMessage("You should ensure that the version of python where ",
                          "tensorflow is installed is either the default python ",
                          "on the system PATH or is specified explicitly via the ",
                          "TENSORFLOW_PYTHON environment variable.\n")
    if (!is.null(.tf_config)) {
      packageStartupMessage("Detected Python configuration:\n")
      packageStartupMessage(str(.tf_config))
    }
  }
}

.onUnload <- function(libpath) {
  py_finalize();
}

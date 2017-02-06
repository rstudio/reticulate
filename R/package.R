
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

# package level mutable global state
.globals <- new.env(parent = emptyenv())
.globals$py_config <- NULL
.globals$load_error_message <- NULL

.onLoad <- function(libname, pkgname) {

  # attempt to load tensorflow
  tf <<- tryCatch(import("tensorflow"), error = function(e) e)
  if (inherits(tf, "error")) {
    .globals$load_error_message <- tf$message
    tf <<- NULL
    return()
  }

  # if we loaded tensorflow then register tf help topics
  register_tf_help_topics()
}


.onAttach <- function(libname, pkgname) {

  if (is.null(tf)) {
    packageStartupMessage("\n", .globals$load_error_message)
    packageStartupMessage("\nIf you have not yet installed TensorFlow, see ",
                          "https://www.tensorflow.org/get_started/\n")
    packageStartupMessage("You should ensure that the version of python where ",
                          "tensorflow is installed is either the default python ",
                          "on the system PATH or is specified explicitly via the ",
                          "TENSORFLOW_PYTHON environment variable.\n")
    if (!is.null(.globals$py_config)) {
      packageStartupMessage("Detected Python configuration:\n")
      packageStartupMessage(str(.globals$py_config))
    }
  }
}

.onUnload <- function(libpath) {
  py_finalize();
}


is_python_initialized <- function() {
  !is.null(.globals$py_config)
}


ensure_python_initialized <- function(required_module = NULL) {
  if (!is_python_initialized())
    .globals$py_config <- initialize_python(required_module)
}

initialize_python <- function(required_module = NULL) {

  # find configuration
  config <- py_discover_config(required_module)

  # check for basic python prerequsities
  if (is.null(config)) {
    stop("Installation of Python not found, Python bindings not loaded.")
  } else if (!file.exists(config$libpython)) {
    stop("Python shared library '", config$libpython, "' not found, Python bindings not loaded.")
  } else if (is.null(config$numpy)) {
    stop("Installation of Numpy not found, Python bindings not loaded.")
  } else if (is_incompatible_arch(config)) {
    stop("Your current architecture is ", python_arch(), " however this version of ",
         "Python is compiled for ", config$architecture, ".")
  }

  # initialize python
  py_initialize(config$python,
                config$libpython,
                config$pythonhome,
                config$virtualenv_activate,
                config$version >= "3.0");

  # set available flag indicating we have py bindings
  config$available <- TRUE

  # add our python scripts to the search path
  py_run_string(paste0("import sys; sys.path.append('",
                       system.file("python", package = "tensorflow") ,
                       "')"))

  # return config
  config
}





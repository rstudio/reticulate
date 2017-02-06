
#' R interface to Python
#'
#' @docType package
#' @name rpy
#' @useDynLib rpy
#' @importFrom Rcpp evalCpp
NULL

# package level mutable global state
.globals <- new.env(parent = emptyenv())
.globals$py_config <- NULL
.globals$load_error_message <- NULL
.globals$suppress_warnings_handlers <- list()



.onUnload <- function(libpath) {
  if (is_python_initialized())
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
  } else if (config$numpy$version < "1.11") {
    stop("Installation of Numpy >= 1.11 not found, Python bindings not loaded.")
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
                       system.file("python", package = "rpy") ,
                       "')"))

  # return config
  config
}





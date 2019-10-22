
#' R Interface to Python
#'
#' R interface to Python modules, classes, and functions. When calling into
#' Python R data types are automatically converted to their equivalent Python
#' types. When values are returned from Python to R they are converted back to R
#' types. The reticulate package is compatible with all versions of Python >= 2.7.
#' Integration with NumPy requires NumPy version 1.6 or higher.
#'
#' @docType package
#' @name reticulate
#' @useDynLib reticulate, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

# package level mutable global state
.globals <- new.env(parent = emptyenv())
.globals$required_python_version <- NULL
.globals$use_python_versions <- c()
.globals$py_config <- NULL
.globals$delay_load_module <- NULL
.globals$delay_load_environment <- NULL
.globals$delay_load_priority <- 0
.globals$suppress_warnings_handlers <- list()
.globals$class_filters <- list()
.globals$py_repl_active <- FALSE


is_python_initialized <- function() {
  !is.null(.globals$py_config)
}


ensure_python_initialized <- function(required_module = NULL) {

  if (!is_python_initialized()) {
    # give delay load modules priority
    use_environment <- NULL
    if (!is.null(.globals$delay_load_module)) {
      required_module <- .globals$delay_load_module
      use_environment <- .globals$delay_load_environment
      .globals$delay_load_module <- NULL # one shot
      .globals$delay_load_environment <- NULL
      .globals$delay_load_priority <- 0
    }

    .globals$py_config <- initialize_python(required_module, use_environment)

    # generate 'R' helper object
    py_inject_r(envir = globalenv())

    # remap output streams to R output handlers
    remap_output_streams()
    
    # inject hooks
    py_inject_hooks()
    
    # install required packages
    configure_environment()

  }
}

initialize_python <- function(required_module = NULL, use_environment = NULL) {

  # provide hint to install Miniconda if no Python is found
  python_not_found <- function(msg) {
    hint <- "Use reticulate::install_miniconda() if you'd like to install a Miniconda Python environment."
    stop(paste(msg, hint, sep = "\n"), call. = FALSE)
  }

  # resolve top level module for search
  if (!is.null(required_module))
    required_module <- strsplit(required_module, ".", fixed = TRUE)[[1]][[1]]

  # find configuration
  config <- py_discover_config(required_module, use_environment)

  # check for basic python prerequsities
  if (is.null(config)) {
    python_not_found("Installation of Python not found, Python bindings not loaded.")
  } else if (!is_windows() && is.null(config$libpython)) {
    python_not_found("Python shared library not found, Python bindings not loaded.")
  } else if (is_incompatible_arch(config)) {
    fmt <- "Your current architecture is %s; however, this version of Python was compiled for %s."
    msg <- sprintf(fmt, current_python_arch(), config$architecture)
    python_not_found(msg)
  }

  # check numpy version and provide a load error message if we don't satisfy it
  if (is.null(config$numpy) || config$numpy$version < "1.6")
    numpy_load_error <- "installation of Numpy >= 1.6 not found"
  else
    numpy_load_error <- ""

  # if we're a virtual environment then set VIRTUAL_ENV (need to
  # set this before initializing Python so that module paths are
  # set as appropriate)
  if (nzchar(config$virtualenv))
    Sys.setenv(VIRTUAL_ENV = config$virtualenv)

  # set R_SESSION_INITIALIZED flag (used by rpy2)
  curr_session_env <- Sys.getenv("R_SESSION_INITIALIZED", unset = NA)
  Sys.setenv(R_SESSION_INITIALIZED = sprintf('PID=%s:NAME="reticulate"', Sys.getpid()))

  # initialize python
  oldpath <- python_munge_path(config$python)
  tryCatch(
    {
      py_initialize(config$python,
                    config$libpython,
                    config$pythonhome,
                    config$virtualenv_activate,
                    config$version >= "3.0",
                    interactive(),
                    numpy_load_error)
    },
    error = function(e) {
      Sys.setenv(PATH = oldpath)
      if (is.na(curr_session_env)) {
        Sys.unsetenv("R_SESSION_INITIALIZED")
      } else {
        Sys.setenv(R_SESSION_INITIALIZED = curr_session_env)
      }
      stop(e)
    }
  )

  # set available flag indicating we have py bindings
  config$available <- TRUE

  # add our python scripts to the search path
  py_run_string_impl(paste0("import sys; sys.path.append('",
                       system.file("python", package = "reticulate") ,
                       "')"))

  # ensure modules can be imported from the current working directory
  py_run_string_impl("import sys; sys.path.insert(0, '')")

  # notify the user if the loaded version of Python isn't the same
  # as the requested version of python
  local({

    # nothing to do if user didn't request any version
    requested_versions <- reticulate_python_versions()
    if (length(requested_versions) == 0)
      return()

    # if we loaded one of the requested versions, everything is ok
    if (config$python %in% requested_versions)
      return()

    # otherwise, warn that we were unable to honor their request
    if (length(requested_versions) == 1) {
      fmt <- paste(
        "Python '%s' was requested but '%s' was loaded instead",
        "(see reticulate::py_config() for more information)"
      )
      msg <- sprintf(fmt, requested_versions[[1]], config$python)
      warning(msg, call. = FALSE)
    } else {
      fmt <- paste(
        "could not honor request to load desired versions of Python; '%s' was loaded instead",
        "(see reticulate::py_config() for more information)"
      )
      msg <- sprintf(fmt, config$python)
      warning(msg, call. = FALSE)
    }

  })

  # return config
  config
}




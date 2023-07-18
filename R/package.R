
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
#' @keywords internal
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
.globals$class_filters <- list(

  function(classes) {
    if ("python.builtin.BaseException" %in% classes) {
      classes <- unique(c(classes, "error", "condition"))
    }
    classes
  }

)
.globals$py_repl_active <- FALSE

is_python_initialized <- function() {
  !is.null(.globals$py_config)
}


ensure_python_initialized <- function(required_module = NULL) {

  # nothing to do if python is initialized
  if (is_python_initialized())
    return()

  # give delay load modules priority
  use_environment <- NULL
  if (!is.null(.globals$delay_load_module)) {
    required_module <- .globals$delay_load_module
    use_environment <- .globals$delay_load_environment
    .globals$delay_load_module <- NULL # one shot
    .globals$delay_load_environment <- NULL
    .globals$delay_load_priority <- 0
  }

  # notify front-end (if any) that Python is about to be initialized
  callback <- getOption("reticulate.python.beforeInitialized")
  if (is.function(callback))
    callback()

  # perform initialization
  .globals$py_config <- initialize_python(required_module, use_environment)

  # remap output streams to R output handlers
  remap_output_streams()
  set_knitr_python_stdout_hook()

  # generate 'R' helper object
  py_inject_r()

  # inject hooks
  py_inject_hooks()

  # install required packages
  configure_environment()

  # notify front-end (if any) that Python has been initialized
  callback <- getOption("reticulate.python.afterInitialized")
  if (is.null(callback))
    callback <- getOption("reticulate.initialized")

  if (is.function(callback))
    callback()

  # set up a Python signal handler
  signals <- import("rpytools.signals")
  signals$initialize(py_interrupts_pending)

  # register C-level interrupt handler
  py_register_interrupt_handler()

  # call init hooks
  call_init_hooks()

}


call_init_hooks <- function() {
  for (fun in get_hooks_list("reticulate.onPyInit")) {
    if (is.character(fun)) {
      fun <- get(fun)
    }
    fun()
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
  config <- local({
    op <- options(reticulate.python.initializing = TRUE)
    on.exit(options(op), add = TRUE)
    py_discover_config(required_module, use_environment)
  })

  # check if R is embedded in an python environment
  py_embedded <- !is.null(main_process_python_info())

  # check for basic python pre-requisites
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
  numpy_load_error <- tryCatch(

    expr = {
      if (is.null(config$numpy) || config$numpy$version < "1.6")
        "installation of Numpy >= 1.6 not found"
      else
        ""
    },

    error = function(e) "<unknown>"

  )

  # if we're a virtual environment then set VIRTUAL_ENV (need to
  # set this before initializing Python so that module paths are
  # set as appropriate)
  if (nzchar(config$virtualenv))
    Sys.setenv(VIRTUAL_ENV = config$virtualenv)

  # set R_SESSION_INITIALIZED flag (used by rpy2)
  curr_session_env <- Sys.getenv("R_SESSION_INITIALIZED", unset = NA)
  Sys.setenv(R_SESSION_INITIALIZED = sprintf('PID=%s:NAME="reticulate"', Sys.getpid()))

  # prefer utf-8 encoding on Windows in RStudio
  if (is_rstudio()) {
    encoding <- Sys.getenv("PYTHONIOENCODING", unset = NA)
    if (is.na(encoding))
      Sys.setenv(PYTHONIOENCODING = "utf-8")
  }

  # munge PATH for python (needed so libraries can be found in some cases)
  oldpath <- python_munge_path(config$python)

  # on macOS, we need to do some gymnastics to ensure that Anaconda
  # libraries can be properly discovered (and this will only work in RStudio)
  if (is_osx()) local({

    symlink <- Sys.getenv("RSTUDIO_FALLBACK_LIBRARY_PATH", unset = NA)
    if (is.na(symlink))
      return()

    unlink(symlink)
    target <- dirname(config$libpython)
    file.symlink(target, symlink)

  })

  # initialize python
  tryCatch({

    # set PYTHONPATH (required to load virtual environments in some cases?)
    oldpythonpath <- Sys.getenv("PYTHONPATH")
    newpythonpath <- Sys.getenv(
      "RETICULATE_PYTHONPATH",
      unset = paste(
        config$pythonpath,
        system.file("python", package = "reticulate"),
        sep = .Platform$path.sep
      )
    )

    local({

      # set PYTHONPATH while we initialize
      Sys.setenv(PYTHONPATH = newpythonpath)
      on.exit(Sys.setenv(PYTHONPATH = oldpythonpath), add = TRUE)

      # initialize Python
      py_initialize(config$python,
                    config$libpython,
                    config$pythonhome,
                    config$virtualenv_activate,
                    config$version >= "3.0",
                    interactive(),
                    numpy_load_error)

    })

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

  if (py_embedded) {

    # we need to insert path to rpytools directly for embedded R
    path <- system.file("python", package = "reticulate")
    fmt <- "import sys; sys.path.append(%s)"
    cmd <- sprintf(fmt, shQuote(path))

    py_run_string_impl(cmd)

  }

  if (is_windows()) {
    # patch sys.executable to point to python.exe, not Rterm.exe or rsession-utf8.exe, #1258
    py_run_string_impl("import sys; sys.executable = sys.argv[0]", local = TRUE)
  }

  # ensure modules can be imported from the current working directory
  py_run_string_impl("import sys; sys.path.insert(0, '')", local = TRUE)

  # if this is a conda installation, set QT_QPA_PLATFORM_PLUGIN_PATH
  # https://github.com/rstudio/reticulate/issues/586
  py_set_qt_qpa_platform_plugin_path(config)

  # notify the user if the loaded version of Python isn't the same
  # as the requested version of python
  local({

    # nothing to do if user didn't request any version
    requested_versions <- reticulate_python_versions()
    if (length(requested_versions) == 0)
      return()

    # if we loaded one of the requested versions, everything is ok
    actual <- normalizePath(config$python, winslash = "/", mustWork = FALSE)
    requested <- normalizePath(requested_versions, winslash = "/", mustWork = FALSE)
    if (actual %in% requested)
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

check_forbidden_initialization <- function() {

  if (is_python_initialized())
    return(FALSE)

  override <- getOption(
    "reticulate.allow.package.initialization",
    default = FALSE
  )

  if (identical(override, TRUE))
    return(FALSE)

  calls <- sys.calls()
  frames <- sys.frames()

  for (i in seq_along(calls)) {

    call <- calls[[i]]
    frame <- frames[[i]]
    if (!identical(call[[1]], as.name("runHook")))
      next

    bad <-
      identical(call[[2]], ".onLoad") ||
      identical(call[[2]], ".onAttach")

    if (!bad)
      next

    pkgname <- tryCatch(
        get("pkgname", envir = frame),
        error = function(e) "<unknown>"
      )

    fmt <- paste(
      "package '%s' attempted to initialize Python in %s().",
      "Packages should not initialize Python themselves; rather, Python should",
      "be loaded on-demand as requested by the user of the package. Please see",
      "vignette(\"python_dependencies\", package = \"reticulate\") for more details."
    )

    msg <- sprintf(fmt, pkgname, call[[2]])
    warning(msg)

  }

}

check_forbidden_install <- function(label) {

  # escape hatch for users who know, or claim to know, what they're doing
  envvar <- Sys.getenv("_RETICULATE_I_KNOW_WHAT_IM_DOING_", unset = NA)
  if (identical(tolower(envvar), "true"))
    return(FALSE)

  # if this is being called as part of R CMD check, then warn
  # (error in future versions)
  if (is_r_cmd_check()) {
    fmt <- "cannot install %s during R CMD check"
    msg <- sprintf(fmt, label)
    warning(msg)
    return(TRUE)
  }

  FALSE

}

py_set_qt_qpa_platform_plugin_path <- function(config) {

  # only done on Windows since that's where we see all the issues
  if (!is_windows())
    return(FALSE)

  # get python homes (note that multiple homes may be specified)
  homes <- strsplit(config$pythonhome, ";", fixed = TRUE)[[1]]
  for (home in homes) {

    # build some candidate paths to the plugins directory
    candidates <- c(
      file.path(home, "Library/plugins/platforms"),
      file.path(home, "../../Library/plugins/platforms")
    )

    # check and see if any of these exist
    paths <- candidates[file.exists(candidates)]
    if (length(paths) == 0)
      next

    # we found a path; use it
    path <- normalizePath(paths[[1]], winslash = "/", mustWork = TRUE)
    path <- gsub("/", "\\\\", path, fixed = TRUE)

    fmt <- "import os; os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '%s'"
    cmd <- sprintf(fmt, path)
    py_run_string_impl(cmd)

    # return TRUE to indicate success
    return(TRUE)

  }

  # failed to find folder; nothing to do
  FALSE

}

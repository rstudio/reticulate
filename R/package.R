
#' R Interface to Python
#'
#' R interface to Python modules, classes, and functions. When calling into
#' Python R data types are automatically converted to their equivalent Python
#' types. When values are returned from Python to R they are converted back to R
#' types. The reticulate package is compatible with all versions of Python >= 3.6.
#' Integration with NumPy requires NumPy version 1.6 or higher.
#'
#' @name reticulate
#' @aliases reticulate-package
#' @keywords internal
#' @useDynLib reticulate, .registration = TRUE
#' @importFrom Rcpp evalCpp
"_PACKAGE"

# package level mutable global state
.globals <- new.env(parent = emptyenv())
.globals$required_python_version <- NULL
.globals$use_python_versions <- c()
.globals$py_config <- NULL
.globals$delay_load_imports <- data.frame(module = character(),
                                          priority = integer(),
                                          environment = character(),
                                          stringsAsFactors = FALSE)
.globals$suppress_warnings_handlers <- list()
.globals$class_filters <- list()
.globals$py_repl_active <- FALSE

is_python_initialized <- function() {
  !is.null(.globals$py_config)
}

is_epheremal_venv_initialized <- function() {
  isTRUE(.globals$py_config$ephemeral)
}

is_python_finalized <- function() {
  identical(.globals$finalized, TRUE)
}

ensure_python_initialized <- function(required_module = NULL) {

  # nothing to do if python is initialized
  if (is_python_initialized())
    return()

  if (is_python_finalized())
    stop("py_initialize() cannot be called more than once per R session or after py_finalize(). Please start a new R session.")

  # notify front-end (if any) that Python is about to be initialized
  callback <- getOption("reticulate.python.beforeInitialized")
  if (is.function(callback))
    callback()

  # make sure this module is used for an environment name.
  if(!is.null(required_module))
    register_delay_load_import(required_module)

  # perform initialization
  .globals$py_config <- initialize_python()

  # clear the global list of delay_load requests
  .globals$delay_load_imports <- NULL

  # remap output streams to R output handlers
  remap_output_streams()
  set_knitr_python_stdout_hook()

  if (is_windows() && ( is_rstudio() || is_positron() ))
    import("rpytools.subprocess")$patch_subprocess_Popen()

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

  # re-install interrupt handler -- note that RStudio tries to re-install its
  # own interrupt handler when reticulate is initialized, but reticulate needs
  # to handle interrupts itself (and it can do so compatibly with RStudio)
  install_interrupt_handlers()

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
    hint <- 'See the Python "Order of Discovery" here: https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery.'
    stop(paste(msg, hint, sep = "\n"), call. = FALSE)
  }

  # resolve top level module for search
  if (!is.null(required_module))
    required_module <- strsplit(required_module, ".", fixed = TRUE)[[1L]][[1L]]

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
  # also munge LD_LIBRARY_PATH on Linux
  # (needed for Python 3.12 preinstalled on GHA runners, perhaps other installations too)
  prefix_python_lib_to_ld_library_path(config$python)

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
                    config$version$major,
                    config$version$minor,
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

  # allow enabling the Python finalizer
  reg.finalizer(.globals, function(e) {
    try(py_allow_threads_impl(FALSE))
    if (tolower(Sys.getenv("RETICULATE_ENABLE_PYTHON_FINALIZER")) %in% c("true", "1", "yes"))
      py_finalize()
  }, onexit = TRUE)

  # set available flag indicating we have py bindings
  config$available <- TRUE

  if (py_embedded) {

    # we need to insert path to rpytools directly for embedded R
    path <- system.file("python", package = "reticulate")
    fmt <- "import sys; sys.path.append(%s)"
    cmd <- sprintf(fmt, shQuote(path))

    py_run_string_impl(cmd)

  }

  local({
    # patch sys.executable to point to python.exe, not Rterm.exe or rsession-utf8.exe, #1258
    patch <- sprintf("import sys; sys.executable  = r'''%s'''",
                     config$executable)
    py_run_string_impl(patch, local = TRUE)
  })

  if (nzchar(config$base_executable)) local({
    # just like sys.executable, patch to point to python.exe, not Rterm.exe
    # need to patch for multiprocessing to work on windows, perhaps other things too.
    # in venvs, _base_executable should point to the venv starter, #1430
    patch <- sprintf("import sys; sys._base_executable = r'''%s'''",
                     config$base_executable)
    py_run_string_impl(patch, local = TRUE)
  })

  # ensure modules can be imported from the current working directory
  py_run_string_impl("import sys; sys.path.insert(0, '')", local = TRUE)

  # if this is a conda installation, set QT_QPA_PLATFORM_PLUGIN_PATH
  # https://github.com/rstudio/reticulate/issues/586
  py_set_qt_qpa_platform_plugin_path(config)

  if (was_python_initialized_by_reticulate()) {
    allow_threads <- Sys.getenv("RETICULATE_ALLOW_THREADS", "true")
    allow_threads <- tolower(allow_threads) %in% c("true", "1", "yes")
    if (allow_threads) {
      py_allow_threads_impl(TRUE)
    }
  }

  # return config
  config
}

# unused presently, formerly called from initialize_python()
# https://github.com/rstudio/reticulate/commit/e8c82a1f95eb97c4e5fc27b6550a4498827438e0#r122213856
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

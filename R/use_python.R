
#' Use Python
#'
#' Select the version of Python to be used by `reticulate`.
#'
#' The `reticulate` package initializes its Python bindings lazily -- that is,
#' it does not initialize its Python bindings until an API that explicitly
#' requires Python to be loaded is called. This allows users and package authors
#' to request particular versions of Python by calling `use_python()` or one of
#' the other helper functions documented in this help file.
#'
#' @section RETICULATE_PYTHON:
#'
#' The `RETICULATE_PYTHON` environment variable can also be used to control
#' which copy of Python `reticulate` chooses to bind to. It should be set to
#' the path to a Python interpreter, and that interpreter can either be:
#'
#' - A standalone system interpreter,
#' - Part of a virtual environment,
#' - Part of a Conda environment.
#'
#' When set, this will override any other requests to use a particular copy of
#' Python. Setting this in `~/.Renviron` (or optionally, a project `.Renviron`)
#' can be a useful way of forcing `reticulate` to use a particular version of
#' Python.
#'
#' @section Caveats:
#'
#' Note that the requests for a particular version of Python via `use_python()`
#' and friends only persist for the active session; they must be re-run in each
#' new \R session as appropriate.
#'
#' If `use_python()` (or one of the other `use_*()` functions) are called
#' multiple times, the most recently-requested version of Python will be
#' used. Note that any request to `use_python()` will always be overridden
#' by the `RETICULATE_PYTHON` environment variable, if set.
#'
#' The [py_config()] function will also provide a short note describing why
#' `reticulate` chose to select the version of Python that was ultimately
#' activated.
#'
#' @param python
#'   The path to a Python binary.
#'
#' @param version
#'   The version of Python to use. `reticulate` will search for versions of
#'   Python as installed by the [install_python()] helper function.
#'
#' @param virtualenv
#'   Either the name of, or the path to, a Python virtual environment.
#'
#' @param condaenv
#' The conda environment to use. For `use_condaenv()`, this can be the name,
#' the absolute prefix path, or the absolute path to the python binary. If
#' the name is ambiguous, the first environment is used and a warning is
#' issued. For `use_miniconda()`, the only conda installation searched is
#' the one installed by `install_miniconda()`.
#'
#' @param conda
#'   The path to a `conda` executable. By default, `reticulate` will check the
#'   `PATH`, as well as other standard locations for Anaconda installations.
#'
#' @param required
#'   Is the requested copy of Python required? If `TRUE`, an error will be
#'   emitted if the requested copy of Python does not exist. If `FALSE`, the
#'   request is taken as a hint only, and scanning for other versions will still
#'   proceed. A value of `NULL` (the default), is equivalent to `TRUE`.
#'
#' @importFrom utils file_test
#'
#' @export
use_python <- function(python, required = NULL) {

  required <- required %||% use_python_required()
  if (required && !file_test("-f", python) && !file_test("-d", python))
    stop("Specified version of python '", python, "' does not exist.")

  # ensure that the python path is normalized as expected
  python <- normalize_python_path(python)$path

  # if required == TRUE and python is already initialized then confirm that we
  # are using the correct version
  if (required && is_python_initialized()) {

    if (!file_same(py_config()$python, python)) {

      fmt <- paste(
        "ERROR: The requested version of Python ('%s') cannot be used, as",
        "another version of Python ('%s') has already been initialized.",
        "Please restart the R session if you need to attach reticulate",
        "to a different version of Python."
      )

      msg <- sprintf(fmt, python, py_config()$python)
      writeLines(strwrap(msg), con = stderr())

      stop("failed to initialize requested version of Python")

    }
  }

  # validate that this copy of python has libpython available
  if (required) {
    config <- python_config(python)
    if (is.null(config$libpython)) {

      fmt <- heredoc("
        '%s' was not built with a shared library.
        reticulate can only bind to copies of Python built with '--enable-shared'.
      ")

      stopf(fmt, python)
    }
  }

  if (required) {
    # warn if we're overriding a previous request that will be ignored.
    prior_required_python <- .globals$required_python_version
    if (!is.null(prior_required_python) &&
        !isTRUE(canonical_path(prior_required_python) == canonical_path(python)))
      warningf(heredoc(c(
        'Previous request to `use_python("%s", required = TRUE)` will be ignored.',
        'It is superseded by request to `use_python("%s")')),
        prior_required_python, python)

    .globals$required_python_version <- python

    # warn if this setting will be ignored because RETICULATE_PYTHON is set
    python_w_precedence <- Sys.getenv("RETICULATE_PYTHON", NA)
    if (!is.na(python_w_precedence) &&
        !isTRUE(canonical_path(python_w_precedence) == canonical_path(python)))
      warningf(heredoc(c(
        'The request to `use_python("%s")` will be ignored because the',
        'environment variable RETICULATE_PYTHON is set to "%s"')),
        python, python_w_precedence)
  }

  .globals$use_python_versions <- unique(c(.globals$use_python_versions, python))

}

#' @rdname use_python
#' @export
use_python_version <- function(version, required = NULL) {
  required <- required %||% use_python_required()
  path <- pyenv_python(version)
  use_python(path, required = required)
}

#' @rdname use_python
#' @export
use_virtualenv <- function(virtualenv = NULL, required = NULL) {

  required <- required %||% use_python_required()

  # resolve path to virtualenv
  virtualenv <- virtualenv_path(virtualenv)

  # validate it if required
  if (required && !is_virtualenv(virtualenv))
    stop("Directory ", virtualenv, " is not a Python virtualenv")

  # get path to Python binary
  suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
  python <- file.path(virtualenv, suffix)
  use_python(python, required = required)

}

#' @rdname use_python
#' @export
use_condaenv <- function(condaenv = NULL, conda = "auto", required = NULL) {

  required <- required %||% use_python_required()

  if (grepl("[/\\]", condaenv) &&
      grepl("^python[0-9.]*(exe)?", basename(condaenv))) {
    # path to python binary provided
    python <- condaenv
    if (!is_conda_python(python))
      warning("path supplied to use_condaenv() is not a conda python:\n\t",
              python)
    use_python(python, required = required)
    return(invisible(NULL))
  }

  # check for condaenv supplied by path
  condaenv <- condaenv_resolve(condaenv)
  if (grepl("[/\\]", condaenv) && is_condaenv(condaenv)) {
    python <- conda_python(condaenv)
    use_python(python, required = required)
    return(invisible(NULL))
  }

  # if the user has requested the 'base' environment, then just activate
  # the conda installation associated with the conda binary found
  #
  # TODO: what if there are multiple conda installations? users could still
  # use 'use_python()' explicitly to target a specific install
  conda <- tryCatch(conda_binary(conda), error = identity)
  if (inherits(conda, "error")) {
    if (required)
      stop(conda)
    else
      return(invisible(NULL))
  }

  if (identical(condaenv, "base")) {
    bin <- dirname(conda)
    suffix <- if (is_windows()) "../python.exe" else "python"
    python <- file.path(bin, suffix)
    return(use_python(python, required = required))
  }

  # list all conda environments
  conda_envs <- conda_list(conda)

  # look for one with that name
  matches <- which(conda_envs$name == condaenv)

  # if we had no matches, then either fail or return early as appropriate
  if (length(matches) == 0) {
    if (required)
      stop("Unable to locate conda environment '", condaenv, "'.")
    return(invisible(NULL))
  }

  # check for multiple matches (this could happen if the user has multiple
  # Conda installations, or multiple environment paths)
  envs <- conda_envs[matches, ]
  if (nrow(envs) > 1) {
    output <- paste(capture.output(print(envs)), collapse = "\n")
    warning("multiple Conda environments found; the first-listed will be chosen.\n", output)
  }

  # we now have a copy of Python to use -- add it to the list
  python <- envs$python[[1]]
  use_python(python, required = required)

  invisible(NULL)

}

#' @rdname use_python
#' @export
use_miniconda <- function(condaenv = NULL, required = NULL) {

  required <- required %||% use_python_required()

  # check that Miniconda is installed
  if (!miniconda_exists()) {

    msg <- paste(
      "Miniconda is not installed.",
      "Use reticulate::install_miniconda() to install Miniconda.",
      sep = "\n"
    )
    stop(msg)

  }

  # use it
  use_condaenv(
    condaenv = condaenv,
    conda = miniconda_conda(),
    required = required
  )

}

use_python_required <- function() {

  # for now, assume that calls from within a package's .onLoad() are
  # advisory, but we may consider relaxing this in the future
  calls <- sys.calls()
  for (call in calls) {

    match <-
      length(call) >= 2 &&
      identical(call[[1L]], as.name("runHook")) &&
      identical(call[[2L]], ".onLoad")

    if (match)
      return(FALSE)

  }

  # default to TRUE
  TRUE

}

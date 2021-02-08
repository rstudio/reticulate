
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
#' - A standalone system interprter,
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
#' By default, requests are _advisory_, and may be ignored for a number of reasons:
#' 
#' - The requested copy of Python cannot be initialized,
#' - The requested copy of Python does not have an installation of `numpy` available,
#' - Another call to `use_python()` has requested a different version of Python,
#' - The request has been overridden via `use_python(..., required = TRUE)`.
#' 
#' In general, if you explicitly want to use a particular version of Python, it
#' is recommended to set `required = TRUE`.
#' 
#' Note that the requests for a particular version of Python via `use_python()`
#' and friends only persist for the active session; they must be re-run in each
#' new \R session as appropriate.
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
#'   The name of the Conda environment to use.
#' 
#' @param conda
#'   The path to a `conda` executable. By default, `reticulate` will check the
#'   `PATH`, as well as other standard locations for Anaconda installations.
#'   
#' @param required
#'   Is the requested copy of Python required? If `TRUE`, an error will be
#'   emitted if the requested copy of Python does not exist. Otherwise, the
#'   request is taken as a hint only, and scanning for other versions will still
#'   proceed.
#'
#' @importFrom utils file_test
#'
#' @export
use_python <- function(python, required = FALSE) {

  if (required && !file_test("-f", python) && !file_test("-d", python))
    stop("Specified version of python '", python, "' does not exist.")

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

  if (required)
    .globals$required_python_version <- python

  .globals$use_python_versions <- unique(c(.globals$use_python_versions, python))
}

#' @rdname use_python
#' @export
use_python_version <- function(version, required = FALSE) {
  path <- pyenv_python(version)
  use_python(path, required = required)
}

#' @rdname use_python
#' @export
use_virtualenv <- function(virtualenv = NULL, required = FALSE) {

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
use_condaenv <- function(condaenv = NULL, conda = "auto", required = FALSE) {

  # check for condaenv supplied by path
  condaenv <- condaenv_resolve(condaenv)
  if (grepl("[/\\]", condaenv) && is_condaenv(condaenv)) {
    python <- conda_python(condaenv)
    use_python(python, required = required)
    return(invisible(NULL))
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
use_miniconda <- function(condaenv = NULL, required = FALSE) {
  
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

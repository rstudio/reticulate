
#' Configure which version of Python to use
#'
#' @param python Path to Python binary
#' @param virtualenv Directory of Python virtualenv
#' @param condaenv Name of Conda environment
#' @param conda Conda executable. Default is `"auto"`, which checks the `PATH`
#'   as well as other standard locations for Anaconda installations.
#' @param required Is this version of Python required? If `TRUE` then an error
#'   occurs if it's not located. Otherwise, the version is taken as a hint only
#'   and scanning for other versions will still proceed.
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
  if (grepl("[/\\]", condaenv, fixed = TRUE) && is_condaenv(condaenv)) {
    python <- conda_python(condaenv)
    use_python(python, required = required)
    return(invisible(NULL))
  }

  # list all conda environments
  conda_envs <- conda_list(conda)

  # look for one with that name
  conda_env_python <- subset(conda_envs, conda_envs$name == condaenv)$python
  if (length(conda_env_python) == 0 && required)
    stop("Unable to locate conda environment '", condaenv, "'.")

  if (!is.null(condaenv))
    use_python(conda_env_python, required = required)

  invisible(NULL)
}

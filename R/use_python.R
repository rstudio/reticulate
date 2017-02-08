
#' Configure which version of Python to use
#'
#' @param python Path to Python binary
#' @param virtualenv Directory of Python virtualenv
#' @param condaenv Name of Conda environment
#' @param conda Conda executable. Default is `"auto"`, which checks the `PATH`
#'   as well as other standard locations for Anaconda installations.
#' @param required Is this version of Python required? If `TRUE` then
#'  an error occurs if it's not located. Otherwise, the version is taken
#'  as a hint only and scanning for other versions will still proceed.
#'
#' @importFrom utils file_test
#'
#' @export
use_python <- function(python, required = FALSE) {

  if (required && !file_test("-f", python) && !file_test("-d", python))
    stop("Specified version of python '", python, "' does not exist.")

  options(reticulate.python = c(getOption("reticulate.python"), python))
}


#' @rdname use_python
#' @export
use_virtualenv <- function(virtualenv, required = FALSE) {

  # compute the bin dir
  if (!is_windows())
    python_dir <- file.path(virtualenv, "bin")

  # validate it if required
  if (required) {
    if (!file_test("-d", python_dir) ||
        !file_test("-f", file.path(python_dir, "activate_this.py"))) {
      stop("Directory ", virtualenv, " is not a Python virtualenv")
    }
  }

  # set the option
  use_python(file.path(python_dir, "python"))
}

#' @rdname use_python
#' @export
use_condaenv <- function(condaenv, conda = "auto", required = FALSE) {

  # resolve conda binary
  if (identical(conda, "auto")) {
    conda <- Sys.which("conda")
    if (!nzchar(conda)) {
      conda_locations <- c(
        path.expand("~/anaconda/bin/conda"),
        path.expand("~/anaconda3/bin/conda")
      )
      if (is_windows()) {
        anaconda_versions <- read_anaconda_versions_from_registry()
        if (length(anaconda_versions) > 0) {
          conda_scripts <- file.path(dirname(anaconda_versions), "Scripts", "conda.exe")
          conda_locations <- c(conda_locations, conda_scripts)
        }
      }
    }
    conda_locations <- conda_locations[file.exists(conda_locations)]
    if (length(conda_locations) > 0)
      conda <- conda_locations[[1]]
    else if (required)
      stop("Unable to locate conda binary, please specify 'conda' argument explicitly.")
    else
      return(invisible(NULL))
  } else if (!file.exists(conda)) {
      stop("Specified conda binary '", conda, "' does not exist.")
  }

  # use conda to probe for environments
  conda_envs <- system2(conda, args = c("info", "--envs"), stdout = TRUE)
  matches <- regexec(paste0("^",condaenv,"[ \\*]+(.*)$"), conda_envs)
  matches <- regmatches(conda_envs, matches)
  for (match in matches) {
    if (length(match) == 2) {
      conda_env_dir <- match[[2]]
      if (!is_windows())
        conda_env_dir <- file.path(conda_env_dir, "bin")
      conda_env_python <- file.path(conda_env_dir, "python")
      use_python(conda_env_python)
      break
    }
  }

  if (required)
    stop("Unable to locate conda environment '", condaenv, "'.")

  invisible(NULL)
}



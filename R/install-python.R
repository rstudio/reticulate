
#' Install Python
#'
#' Download and install Python, using the [pyenv](https://github.com/pyenv/pyenv).
#' and [pyenv-win](https://github.com/pyenv-win/pyenv-win) projects.
#'
#' In general, it is recommended that Python virtual environments are created
#' using the copies of Python installed by [install_python()]. For example:
#'
#' ```
#' library(reticulate)
#' version <- "3.8"
#'
#' install_python(version = version, envname = "my-environment")
#' # -- or --
#' python <- install_python(version = version, create_virtualenv = FALSE)
#' virtualenv_create("my-environment", python = python)
#'
#' use_virtualenv("my-environment")
#' ```
#'
#' @param version The version of Python to install. If the patch level is
#'   ommitted, then the latest patch level is automatically selected.
#'
#' @param list Boolean; if set, list the set of available Python versions?
#'
#' @param force Boolean; force re-installation even if the requested version
#'   of Python is already installed?
#'
#' @param create_virtualenv Boolean; Automatically create a virtualenv with
#'   the python installation.
#'
#' @param ... Passed on to `[virtualenv_create()]` if `create_virtualenv` is
#'   `TRUE`
#'
#' @export
install_python <- function(version = "3.8",
                           list = FALSE,
                           force = FALSE,
                           create_virtualenv = TRUE,
                           ...)
{
  # resolve pyenv path
  pyenv <- pyenv_find()
  if (!file.exists(pyenv))
    stop("could not locate 'pyenv' binary")

  valid_versions <- pyenv_list(pyenv = pyenv)

  # if list is set, then list available versions instead
  if (identical(list, TRUE))
    return(valid_versions)

  version <- as.character(version)

  if (!(version %in% valid_versions))
    tryCatch({
      # accept versions like "3.8", and automatically select the latest patchlevel.
      # If we error here, just proceed and let pyenv raise the error.
      # We do this manually here until https://github.com/pyenv/pyenv/issues/2145 is resolved.
      valid_versions <-
        valid_versions[startsWith(valid_versions, version)]
      version. <- paste0(version, ".")
      patchlevels <- lapply(valid_versions, function(v)
        suppressWarnings(as.integer(str_drop_prefix(v, version.))))
      version_w_patchlevel <-
        paste0(version., max(unlist(patchlevels), na.rm = TRUE))
      if (version_w_patchlevel %in% valid_versions)
        version <- version_w_patchlevel
    },
    error = function(e) NULL)

  # install the requested package
  status <- pyenv_install(version, force, pyenv = pyenv)
  if (!identical(status, 0L))
    stopf("installation of Python %s failed", version)

  # ensure that virtualenv is installed for older versions of Python
  python <- pyenv_python(version = version)
  version <- python_version(python)
  if (version < "3.5" && !python_has_module(python, "virtualenv"))
    pip_install(python, "virtualenv")

  # return path to python
  if (create_virtualenv)
    virtualenv_create(python = python, ...)
  else
    python
}

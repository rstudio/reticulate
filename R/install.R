
py_install_method_detect <- function(envname, conda = "auto") {

  # on Windows, we always use conda environments
  if (is_windows()) {

    # validate that conda is available
    conda <- tryCatch(conda_binary(conda = conda), error = identity)
    if (inherits(conda, "error")) {
      msg <- paste(
        "Conda installation not found (failed to locate conda binary)",
        "Please install Anaconda 3.x for Windows (https://www.anaconda.com/download/#windows) before proceeding.",
        sep = "\n"
      )
      stop(msg, call. = FALSE)
    }

    # return it
    return("conda")

  }

  # try to find an existing virtualenv
  if (virtualenv_exists(envname))
    return("virtualenv")

  # try to find an existing condaenv
  if (condaenv_exists(envname, conda = conda))
    return("conda")

  # check to see if virtualenv or venv is available
  python <- virtualenv_default_python()
  if (python_has_module(python, "virtualenv") || python_has_module(python, "venv"))
    return("virtualenv")

  # check to see if conda is available
  conda <- tryCatch(conda_binary(conda = conda), error = identity)
  if (!inherits(conda, "error"))
    return("conda")

  # default to virtualenv
  "virtualenv"

}

#' Install Python packages
#'
#' Install Python packages into a virtual environment or Conda environment.
#'
#' @inheritParams conda_install
#'
#' @param packages Character vector with package names to install.
#' @param envname The name, or full path, of the environment in which Python
#'   packages are to be installed.
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on Windows.
#' @param ... Additional arguments passed to [conda_install()]
#'   or [virtualenv_install()].
#'
#' @details On Linux and OS X the "virtualenv" method will be used by default
#'   ("conda" will be used if virtualenv isn't available). On Windows, the
#'   "conda" method is always used.
#'
#' @seealso [conda-tools], [virtualenv-tools]
#'
#' @export
py_install <- function(packages,
                       envname = Sys.getenv("RETICULATE_PYTHON_ENV", unset = "r-reticulate"),
                       method = c("auto", "virtualenv", "conda"),
                       conda = "auto",
                       ...)
{
  # resolve 'auto' method
  method <- match.arg(method)
  if (method == "auto")
    method <- py_install_method_detect(envname = envname, conda = conda)

  # validate method
  if (identical(method, "virtualenv") && is_windows()) {
    stop("Installing Python packages into a virtualenv is not supported on Windows",
         call. = FALSE)
  }

  # perform the install
  switch(
    method,
    virtualenv = virtualenv_install(envname = envname, packages = packages, ...),
    conda = conda_install(envname, packages = packages, conda = conda, ...),
    stop("unrecognized installation method '", method, "'")
  )

  invisible(NULL)

}

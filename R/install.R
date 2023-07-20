
py_install_method_detect <- function(envname, conda = "auto") {

  # try to find an existing virtualenv
  if (virtualenv_exists(envname))
    return("virtualenv")

  # check and prompt for miniconda
  if (miniconda_enabled() && miniconda_installable() && !miniconda_exists())
    miniconda_install_prompt()

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
#' @param packages A vector of Python packages to install.
#'
#' @param envname The name, or full path, of the environment in which Python
#'   packages are to be installed. When `NULL` (the default), the active
#'   environment as set by the `RETICULATE_PYTHON_ENV` variable will be used;
#'   if that is unset, then the `r-reticulate` environment will be used.
#'
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on Windows.
#'
#' @param python_version The requested Python version. Ignored when attempting
#'   to install with a Python virtual environment.
#'
#' @param pip Boolean; use `pip` for package installation? This is only relevant
#'   when Conda environments are used, as otherwise packages will be installed
#'   from the Conda repositories.
#'
#' @param ... Additional arguments passed to [conda_install()]
#'   or [virtualenv_install()].
#'
#' @param pip_ignore_installed,ignore_installed Boolean; whether pip should
#'   ignore previously installed versions of the requested packages. Setting
#'   this to `TRUE` causes pip to install the latest versions of all
#'   dependencies into the requested environment. This ensure that no
#'   dependencies are satisfied by a package that exists either in the site
#'   library or was previously installed from a different--potentially
#'   incompatible--distribution channel. (`ignore_installed` is an alias for
#'   `pip_ignore_installed`, `pip_ignore_installed` takes precedence).
#'
#' @details On Linux and OS X the "virtualenv" method will be used by default
#'   ("conda" will be used if virtualenv isn't available). On Windows, the
#'   "conda" method is always used.
#'
#' @seealso
#'   [conda_install()], for installing packages into conda environments.
#'   [virtualenv_install()], for installing packages into virtual environments.
#'
#' @export
py_install <- function(packages,
                       envname = NULL,
                       method = c("auto", "virtualenv", "conda"),
                       conda = "auto",
                       python_version = NULL,
                       pip = FALSE,
                       ...,
                       pip_ignore_installed = ignore_installed,
                       ignore_installed = FALSE
                       )
{
  check_forbidden_install("Python packages")

  # if 'envname' was not provided, use the 'active' version of Python
  if (is.null(envname)) {

    python <- tryCatch(py_exe(), error = function(e) NULL)
    if (!is.null(python)) {

      # get information on default version of python
      info <- python_info(python)

      # if this version of python is associated with a python environment,
      # then set 'envname' to use that environment
      type <- info$type %||% "unknown"
      if (type %in% c("virtualenv", "conda"))
        envname <- info$root

      # update installation method
      if (identical(info$type, "virtualenv"))
        method <- "virtualenv"
      else if (identical(info$type, "conda"))
        method <- "conda"

      # update conda binary path if required
      if (identical(conda, "auto") && identical(info$type, "conda"))
        conda <- info$conda %||% find_conda()[[1L]]

    }

  }

  # resolve 'auto' method
  method <- match.arg(method)
  if (method == "auto")
    method <- py_install_method_detect(envname = envname, conda = conda)

  # perform the install
  switch(

    method,

    virtualenv = virtualenv_install(
      envname = envname,
      packages = packages,
      ignore_installed = pip_ignore_installed,
      ...
    ),

    conda = conda_install(
      envname,
      packages = packages,
      conda = conda,
      python_version = python_version,
      pip = pip,
      pip_ignore_installed = pip_ignore_installed,
      ...
    ),

    stop("unrecognized installation method '", method, "'")

  )

  invisible(NULL)

}

# given the name of, or path to, a Python virtual environment,
# try to resolve the path to the python executable associated
# with that environment
py_resolve <- function(envname = NULL,
                       type = c("auto", "virtualenv", "conda"))
{
  # if envname was not supplied, then use the 'default' python
  if (is.null(envname))
    return(py_exe())

  type <- match.arg(type)

  # if envname was supplied, try to resolve the environment path
  envpath <- if (type == "virtualenv") {

    envpath <- virtualenv_path(envname)
    if (!file.exists(envpath))
      stopf("Python virtual environment '%s' does not exist", envname)
    envpath

  } else if (type == "conda") {

    envpath <- condaenv_path(envname)
    if (!file.exists(envpath))
      stopf("Python conda environment '%s' does not exist", envname)
    envpath

  } else if (type == "auto") local({

    envpath <- virtualenv_path(envname)
    if (file.exists(envpath))
      return(envpath)

    envpath <- condaenv_path(envname)
    if (file.exists(envpath))
      return(envpath)

    stopf("Python environment '%s' does not exist", envname)

  })

  # resolve the path to python
  info <- python_info(envpath)
  info$python

}


#' List installed Python packages
#'
#' List the Python packages that are installed in the requested Python
#' environment.
#'
#' When `envname` is `NULL`, `reticulate` will use the "default" version
#' of Python, as reported by [py_exe()]. This implies that you
#' can call `py_list_packages()` without arguments in order to list
#' the installed Python packages in the version of Python currently
#' used by `reticulate`.
#'
#' @param envname The name of, or path to, a Python virtual environment.
#'   Ignored when `python` is non-`NULL`.
#'
#' @param type The virtual environment type. Useful if you have both
#'   virtual environments and Conda environments of the same name on
#'   your system, and you need to disambiguate them.
#'
#' @param python The path to a Python executable.
#'
#' @returns An \R data.frame, with columns:
#'
#' \describe{
#' \item{`package`}{The package name.}
#' \item{`version`}{The package version.}
#' \item{`requirement`}{The package requirement.}
#' \item{`channel`}{(Conda only) The channel associated with this package.}
#' }
#'
#' @export
py_list_packages <- function(envname = NULL,
                             type = c("auto", "virtualenv", "conda"),
                             python = NULL)
{
  type <- match.arg(type)
  python <- python %||% py_resolve(envname, type)

  info <- python_info(python)
  if (info$type == "conda")
    return(conda_list_packages(info$root))

  pip_freeze(python)
}

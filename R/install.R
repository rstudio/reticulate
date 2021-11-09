
py_install_method_detect <- function(envname, conda = "auto") {

  # try to find an existing virtualenv
  if (!is_windows() && virtualenv_exists(envname))
    return("virtualenv")
  
  # check and prompt for miniconda
  if (miniconda_enabled() && miniconda_installable() && !miniconda_exists())
    miniconda_install_prompt()

  # try to find an existing condaenv
  if (condaenv_exists(envname, conda = conda))
    return("conda")

  # check to see if virtualenv or venv is available
  python <- virtualenv_default_python()
  if (!is_windows() && (python_has_module(python, "virtualenv") || python_has_module(python, "venv")))
    return("virtualenv")

  # check to see if conda is available
  conda <- tryCatch(conda_binary(conda = conda), error = identity)
  if (!inherits(conda, "error"))
    return("conda")
  
  if (is_windows())
    stop("No conda installation detected.", call. = FALSE)

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
#' @seealso [conda-tools], [virtualenv-tools]
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
  
  # if no environment name was provided,
  # but python has already been initialized,
  # then install into that same environment
  if (is.null(envname)) {
    
    python <- if (is_python_initialized())
      .globals$py_config$python
    else if (length(.globals$required_python_version))
      .globals$required_python_version[[1]]
    else if (length(p <- py_discover_config()$python))
      p
    
    if (!is.null(python)) {
      
      # form path to environment
      info <- python_info(python)
      envname <- info$root
      
      # update conda binary path if required
      if (identical(conda, "auto") && identical(info$type, "conda"))
        conda <- info$conda %||% find_conda()
      
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

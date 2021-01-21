
#' Install Python
#' 
#' Download and install Python.
#' 
#' On Windows, the official binaries as made available from <https://ww.python.org>
#' are used. On macOS and Linux, Python is built from sources via the
#' [pyenv](https://github.com/pyenv/pyenv) module.
#' 
#' @param version The version of Python to install.
#' 
#' @param list Boolean; if set, list the set of available Python versions?
#' 
#' @param force Boolean; force re-installation even if the requested version
#'   of Python is already installed?
#' 
install_python <- function(version,
                           list = FALSE,
                           force = FALSE)
{
  # for Windows, we install official binaries
  if (is_windows())
    return(install_python_windows(version))
  
  # check for pyenv on PATH; if it does not exist then install it
  pyenv <- pyenv_find()
  if (!file.exists(pyenv))
    stop("could not locate 'pyenv' binary")
  
  # if list is set, then list available versions instead
  if (identical(list, TRUE))
    return(pyenv_list(pyenv = pyenv))
  
  # install the requested package
  pyenv_install(version, force, pyenv = pyenv)
  
}

install_python_windows <- function(version) {
  
  fmt <- "https://www.python.org/ftp/python/%s/python-%s-amd64.exe"
  url <- sprintf(fmt, version, version)
  # TODO
}

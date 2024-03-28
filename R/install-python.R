
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
#' version <- "3.9.12"
#' install_python(version)
#' virtualenv_create("my-environment", version = version)
#' use_virtualenv("my-environment")
#'
#' # There is also support for a ":latest" suffix to select the latest patch release
#' install_python("3.9:latest") # install latest patch available at python.org
#'
#' # select the latest 3.9.* patch installed locally
#' virtualenv_create("my-environment", version = "3.9:latest")
#' ```
#'
#' @param version The version of Python to install.
#'
#' @param list Boolean; if set, list the set of available Python versions?
#'
#' @param force Boolean; force re-installation even if the requested version
#'   of Python is already installed?
#'
#' @param optimized Boolean; if `TRUE`, installation will take significantly longer but
#'  should result in a faster Python interpreter. Only applicable on macOS and Linux.
#'
#' @note On macOS and Linux this will build Python from sources, which may
#'   take a few minutes. Installation will be faster if some build
#'   dependencies are preinstalled. See
#'   <https://github.com/pyenv/pyenv/wiki#suggested-build-environment> for
#'   example commands you can run to pre-install system dependencies
#'   (requires administrator privileges).
#'
#'  If `optimized = TRUE`, (the default) Python is build with:
#'   ```
#'   PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-lto"
#'   PYTHON_CFLAGS="-march=native -mtune=native"
#'   ```
#'
#'  If `optimized = FALSE`, Python is built with:
#'   ```
#'   PYTHON_CONFIGURE_OPTS=--enable-shared
#'   ```
#'
#'   On Windows, prebuilt installers from <https://www.python.org> are used.
#'
#' @export
install_python <- function(version = "3.9:latest",
                           list = FALSE,
                           force = FALSE,
                           optimized = TRUE)
{

  check_forbidden_install("Python")

  # resolve pyenv path
  pyenv <- pyenv_find()
  if (!file.exists(pyenv))
    stop("could not locate 'pyenv' binary")

  # make sure pyenv is up-to-date
  pyenv_update(pyenv)

  # if list is set, then list available versions instead
  if (identical(list, TRUE))
    return(pyenv_list(pyenv = pyenv))

  if (endsWith(version, ":latest"))
    version <-
      pyenv_resolve_latest_patch(version, installed = FALSE, pyenv = pyenv)

  # install the requested package
  status <- pyenv_install(version, force, pyenv = pyenv, optimized = optimized)
  if (!identical(status, 0L))
    stopf("installation of Python %s failed", version)

  # ensure that virtualenv is installed for older versions of Python
  python <- pyenv_python(version = version)
  version <- python_version(python)
  if (version < "3.5" && !python_has_module(python, "virtualenv"))
    pip_install(python, "virtualenv")

  # return path to python
  python

}

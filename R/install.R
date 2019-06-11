


#' Install Python packages
#'
#' Install Python packages into a virtualenv or conda env.
#'
#' @inheritParams conda_install
#'
#' @param packages Character vector with package names to install
#' @param envname Name of environment to install packages into
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
py_install <- function(
  packages,
  envname = NULL,
  method = c("auto", "virtualenv", "conda"),
  conda = "auto",
  ...) {

  # validate method
  if (identical(method, "virtualenv") && is_windows()) {
    stop("Installing Python packages into a virtualenv is not supported on Windows",
         call. = FALSE)
  }

  # when envname is NULL, use default-supplied environment
  # (or r-reticulate environment otherwise for backwards compatibility)
  if (is.null(envname))
    envname <- Sys.getenv("RETICULATE_PYTHON_ENV", unset = "r-reticulate")

  # find out which methods we can try
  method_available <- function(name) any(method %in% c("auto", name))
  virtualenv_available <- method_available("virtualenv")
  conda_available <- method_available("conda")

  # resolve and look for conda
  conda <- tryCatch(conda_binary(conda), error = function(e) NULL)
  have_conda <- conda_available && !is.null(conda)

  # mac and linux
  if (is_unix()) {

    # check for explicit conda method
    if (identical(method, "conda")) {

      # validate that we have conda
      if (!have_conda)
        stop("Conda installation failed (no conda binary found)\n", call. = FALSE)

      # do install
      conda_install(envname, packages = packages, conda = conda, ...)

    } else {

      # find system python binary
      pyver <- ""
      python <- python_unix_binary("python")
      if (is.null(python)) {
        # try for python3 if we are on linux
        if (is_linux()) {
          python <- python_unix_binary("python3")
          if (is.null(python))
            stop("Unable to locate Python on this system.", call. = FALSE)
          pyver <- "3"
        }
      }
      # find other required tools
      pip <- python_unix_binary(paste0("pip", pyver))
      have_pip <- !is.null(pip)
      virtualenv <- python_unix_binary("virtualenv")
      have_virtualenv <- virtualenv_available && !is.null(virtualenv)

      # if we don't have pip and virtualenv then try for conda if it's allowed
      if ((!have_pip || !have_virtualenv) && have_conda) {

        conda_install(envname, packages = packages, conda = conda, ...)

      # otherwise this is either an "auto" installation w/o working conda
      # or it's an explicit "virtualenv" installation
      } else {

        # validate that we have the required tools for the method
        install_commands <- NULL
        if (is_osx()) {
          if (!have_pip)
            install_commands <- c(install_commands, "$ sudo /usr/bin/easy_install pip")
          if (!have_virtualenv) {
            if (is.null(pip))
              pip <- "/usr/local/bin/pip"
            install_commands <- c(install_commands, sprintf("$ sudo %s install --upgrade virtualenv", pip))
          }
          if (!is.null(install_commands))
            install_commands <- paste(install_commands, collapse = "\n")
        } else if (is_ubuntu()) {
          if (!have_pip) {
            install_commands <- c(install_commands, paste0("$ sudo apt-get install python", pyver ,"-pip"))
            pip <- paste0("/usr/bin/pip", pyver)
          }
          if (!have_virtualenv) {
            if (identical(pyver, "3"))
              install_commands <- c(install_commands, paste("$ sudo", pip, "install virtualenv"))
            else
              install_commands <- c(install_commands, "$ sudo apt-get install python-virtualenv")
          }
          if (!is.null(install_commands))
            install_commands <- paste(install_commands, collapse = "\n")
        } else {
          if (!have_pip)
            install_commands <- c(install_commands, "pip")
          if (!have_virtualenv)
            install_commands <- c(install_commands, "virtualenv")
          if (!is.null(install_commands)) {
            install_commands <- paste("Please install the following Python packages before proceeding:",
                                      paste(install_commands, collapse = ", "))
          }
        }
        if (!is.null(install_commands)) {

          # if these are terminal commands then add special preface
          if (grepl("^\\$ ", install_commands)) {
            install_commands <- paste0(
              "Execute the following at a terminal to install the prerequisites:\n\n",
              install_commands
            )
          }

          stop("Prerequisites for installing Python packages not available.\n\n",
               install_commands, "\n\n", call. = FALSE)
        }

        # do the install
        virtualenv_install(envname, packages, ...)
      }
    }

  # windows installation
  } else {

    # validate that we have conda
    if (!have_conda) {
      stop("Windows Conda installation failed (no conda binary found)\n\n",
           "Install Anaconda 3.x for Windows (https://www.anaconda.com/download/#windows)\n",
           "before proceeding",
           call. = FALSE)
    }

    # do the install
    conda_install(envname, packages, conda = conda, ...)
  }

  cat("\nInstallation complete.\n\n")

  invisible(NULL)
}

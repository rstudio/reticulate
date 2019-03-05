#' Interface to Python Virtual Environments
#'
#' R functions for managing Python [virtual environments](https://virtualenv.pypa.io/en/stable/).
#'
#' Virtual environments are by default located at `~/.virtualenvs` (accessed
#' with the `virtualenv_root` function). You can change the default location by
#' defining defining the `WORKON_HOME` environment variable.
#'
#' Virtual environment functions are not supported on Windows (the use of
#' [conda environments][conda-tools] is recommended on Windows).
#'
#' @param envname The name of, or path to, a Python virtual environment. If
#'   this name contains any slashes, the name will be interpreted as a path;
#'   if the name does not contain slashes, it will be treated as a virtual
#'   environment within `virtualenv_root()`.
#' @param packages A character vector with package names to install or remove.
#' @param ignore_installed Boolean; ignore previously-installed versions of the
#'   requested packages? (This should normally be `TRUE`, so that pre-installed
#'   packages available in the site libraries are ignored and hence packages
#'   are installed into the virtual environment.)
#' @param confirm Boolean; confirm before removing packages or virtual
#'   environments?
#' @param python The path to a Python interpreter, to be used with the created
#'   virtual environment. When `NULL`, the Python interpreter associated with
#'   the current session will be used.
#'
#' @name virtualenv-tools
NULL



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_list <- function() {
  root <- virtualenv_root()
  if (!file.exists(root))
    return(character())
  list.files(root)
}

#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_create <- function(envname, python = NULL) {
  path <- virtualenv_path(envname)

  # check and see if we already have a virtual environment
  if (virtualenv_exists(path)) {
    writeLines(paste("virtualenv:", path))
    return(invisible(path))
  }

  # if the user hasn't requested a particular Python binary,
  # infer from the currently active Python session
  python <- virtualenv_default_python(python)
  writeLines(paste("Creating virtual environment", shQuote(envname), "..."))
  writeLines(paste("Using python:", python))

  # choose appropriate tool for creating virtualenv
  module <- virtualenv_module(python)

  # use it to create the virtual environment
  args <- c("-m", module, "--system-site-packages", path.expand(path))
  result <- system2(python, shQuote(args))
  if (result != 0L) {
    fmt <- "Error creating virtual environment '%s' [error code %d]"
    msg <- sprintf(fmt, envname, result)
    stop(msg, call. = FALSE)
  }

  invisible(path)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_install <- function(envname = NULL, packages, ignore_installed = TRUE) {

  path <- virtualenv_path(envname)
  if (file.exists(path))
    writeLines(paste("Using virtual environment", shQuote(basename(path)), "..."))
  else if (is.null(envname))
    stop("virtualenv_install() called without active virtual environment")
  else
    path <- virtualenv_create(envname)

  # ensure that pip + friends are up-to-date / recent enough
  pip <- virtualenv_pip(path)
  if (pip_version(pip) < "8.1") {
    pip_install(pip, "pip")
    pip_install(pip, "wheel")
    pip_install(pip, "setuptools")
  }

  # now install the requested package
  pip_install(pip, packages, ignore_installed = ignore_installed)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_remove <- function(envname, packages = NULL, confirm = interactive()) {
  path <- virtualenv_path(envname)
  if (!virtualenv_exists(envname)) {
    fmt <- "Virtual environment '%s' does not exist."
    stop(sprintf(fmt, envname), call. = FALSE)
  }

  # packages = NULL means remove the entire virtualenv
  if (is.null(packages)) {

    if (confirm) {
      fmt <- "Remove virtual environment '%s'? [Y/n]: "
      prompt <- sprintf(fmt, envname)
      response <- readline(prompt = prompt)
      if (tolower(response) != "y") {
        writeLines("Operation aborted.")
        return(invisible(NULL))
      }
    }

    unlink(path, recursive = TRUE)
    writeLines(paste("Virtual environment", shQuote(envname), "removed."))
    return(invisible(NULL))

  }

  # otherwise, remove the requested packages
  if (confirm) {
    fmt <- "Remove '%s' from virtual environment '%s'? [Y/n]: "
    prompt <- sprintf(fmt, paste(packages, sep = ", "), path)
    response <- readline(prompt = prompt)
    if (tolower(response) != "y") {
      writeLines("Operation aborted.")
      return(invisible(NULL))
    }
  }

  pip <- virtualenv_pip(path)
  pip_uninstall(pip, packages)
  invisible(NULL)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_root <- function() {
  Sys.getenv("WORKON_HOME", unset = "~/.virtualenvs")
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_python <- function(envname) {
  path.expand(file.path(virtualenv_path(envname), "bin/python"))
}



virtualenv_exists <- function(envname) {
  path <- virtualenv_path(envname)

  # check that the directory exists
  if (!utils::file_test("-d", path))
    return(FALSE)

  # check for some expected files within virtualenv layout
  expected <- c("bin/activate", "bin/pip", "bin/python")
  all(file.exists(file.path(path, expected)))
}



virtualenv_path <- function(envname) {

  # if envname is NULL but Python is already active, then
  # use the path to the current virtual environment
  if (is.null(envname) && py_available()) {
    config <- py_config()
    if (nzchar(config$virtualenv))
      return(config$virtualenv)
  }

  # if envname is still NULL, err
  if (is.null(envname))
    stop("missing environment name")

  # treat environment 'names' containins slashes as paths
  # rather than environments living in WORKON_HOME
  if (grepl("[/\\]", envname)) {
    if (file.exists(envname))
      envname <- normalizePath(envname, winslash = "/")
    return(envname)
  }

  file.path(virtualenv_root(), envname)

}


virtualenv_pip <- function(envname) {
  file.path(virtualenv_path(envname), "bin/pip")
}



virtualenv_default_python <- function(python) {

  if (!is.null(python))
    return(python)

  config <- py_discover_config()
  normalizePath(config$python, winslash = "/")
}



virtualenv_module <- function(python) {
  py_version <- python_version(python)

  # prefer 'venv' for Python 3, but allow for 'virtualenv' for both
  modules <- "virtualenv"
  if (py_version >= "3.6")
    modules <- c("venv", modules)

  # if we have one of thesem odules available, return it
  for (module in modules)
    if (python_has_module(python, module))
      return(module)

  # virtualenv not available: instruct the user to install
  commands <- new_stack()
  commands$push("tools for managing Python virtual environments are not installed.")
  commands$push("")

  # if we don't have pip, recommend its installation
  if (!python_has_module(python, "pip")) {
    commands$push("Install pip with:")
    if (python_has_module(python, "easy_install")) {
      commands$push(paste("$", python, "-m easy_install --upgrade --user pip"))
    } else if (is_ubuntu() && dirname(python) == "/usr/bin") {
      package <- if (py_version < 3) "python-virtualenv" else "python3-venv"
      commands$push(paste("$ sudo apt-get install", package))
    } else {
      commands$push("$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py")
      commands$push(paste("$", python, "get-pip.py"))
    }
    commands$push("")
  }

  # then, recommend installation of virtualenv or ven with pip
  commands$push(paste("Install", modules[[1]], "with:"))
  commands$push(paste("$", python, "-m pip install --upgrade --user", module))

  # report to user
  message <- paste(commands$data(), collapse = "\n")
  stop(message, call. = FALSE)

}



is_python_virtualenv <- function(dir) {

  # virtual environment created with venv
  if (file.exists(file.path(dir, "pyvenv.cfg")))
    return(TRUE)

  # virtual environment created with virtualenv
  if (file.exists(file.path(dir, "bin/activate_this.py")))
    return(TRUE)

  FALSE
}

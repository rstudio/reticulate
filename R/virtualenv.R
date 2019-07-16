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
#'   environment within `virtualenv_root()`. When `NULL`, the virtual environment
#'   as specified by the `RETICULATE_PYTHON_ENV` environment variable will be
#'   used instead.
#'
#' @param packages A character vector with package names to install or remove.
#'
#' @param ignore_installed Boolean; ignore previously-installed versions of the
#'   requested packages? (This should normally be `TRUE`, so that pre-installed
#'   packages available in the site libraries are ignored and hence packages
#'   are installed into the virtual environment.)
#'
#' @param confirm Boolean; confirm before removing packages or virtual
#'   environments?
#'
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
virtualenv_create <- function(envname = NULL, python = NULL) {

  path <- virtualenv_path(envname)
  name <- if (is.null(envname)) path else envname

  # check and see if we already have a virtual environment
  if (virtualenv_exists(path)) {
    writeLines(paste("virtualenv:", name))
    return(invisible(path))
  }

  python <- virtualenv_default_python(python)
  module <- virtualenv_module(python)

  writeLines(paste("Creating virtual environment", shQuote(name), "..."))
  writeLines(paste("Using python:", python))

  # use it to create the virtual environment
  args <- c("-m", module, "--system-site-packages", path.expand(path))
  result <- system2(python, shQuote(args))
  if (result != 0L) {
    fmt <- "Error creating virtual environment '%s' [error code %d]"
    msg <- sprintf(fmt, name, result)
    stop(msg, call. = FALSE)
  }

  # upgrade pip and friends after creating the environment
  # (since the version bundled with virtualenv / venv may be stale)
  pip <- virtualenv_pip(path)
  pip_install(pip, c("pip", "wheel", "setuptools"))

  invisible(path)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_install <- function(envname = NULL, packages, ignore_installed = TRUE) {

  # create virtual environment on demand
  path <- virtualenv_path(envname)
  if (!file.exists(path))
    path <- virtualenv_create(envname)

  # validate that we've received the path to a virtual environment
  name <- if (is.null(envname)) path else envname
  if (!is_virtualenv(path)) {
    fmt <- "'%s' exists but is not a virtual environment"
    stop(sprintf(fmt, name))
  }

  writeLines(paste("Using virtual environment", shQuote(name), "..."))

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
virtualenv_remove <- function(envname = NULL, packages = NULL, confirm = interactive()) {

  path <- virtualenv_path(envname)
  name <- if (is.null(envname)) path else envname

  if (!virtualenv_exists(path)) {
    fmt <- "Virtual environment '%s' does not exist."
    stop(sprintf(fmt, name), call. = FALSE)
  }

  # packages = NULL means remove the entire virtualenv
  if (is.null(packages)) {

    if (confirm) {
      fmt <- "Remove virtual environment '%s'? [Y/n]: "
      prompt <- sprintf(fmt, name)
      response <- readline(prompt = prompt)
      if (tolower(response) != "y") {
        writeLines("Operation aborted.")
        return(invisible(NULL))
      }
    }

    unlink(path, recursive = TRUE)
    writeLines(paste("Virtual environment", shQuote(name), "removed."))
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
virtualenv_python <- function(envname = NULL) {
  path <- virtualenv_path(envname)
  path.expand(file.path(path, "bin/python"))
}



virtualenv_exists <- function(envname = NULL) {

  # try to resolve path
  path <- tryCatch(virtualenv_path(envname), error = identity)
  if (inherits(path, "error"))
    return(FALSE)

  # check that the directory exists
  if (!utils::file_test("-d", path))
    return(FALSE)

  # check for some expected files within virtualenv layout
  expected <- c("bin/activate", "bin/pip", "bin/python")
  all(file.exists(file.path(path, expected)))
}



virtualenv_path <- function(envname = NULL) {

  python_environment_resolve(
    envname = envname,
    resolve = function(envname) file.path(virtualenv_root(), envname)
  )

}


virtualenv_pip <- function(envname) {
  path <- virtualenv_path(envname)
  file.path(path, "bin/pip")
}



virtualenv_default_python <- function(python = NULL) {

  # if the user has supplied a verison of python already, use it
  if (!is.null(python))
    return(python)

  # check for some pre-defined Python sources (prefer Python 3)
  sources <- c(
    Sys.getenv("RETICULATE_PYTHON"),
    .globals$required_python_version,
    Sys.which("python3"),
    Sys.which("python")
  )

  for (source in sources)
    if (nzchar(source) && file.exists(source))
      return(normalizePath(source, winslash = "/"))

  # otherwise, try to explicitly detect Python
  config <- py_discover_config()
  normalizePath(config$python, winslash = "/")

}



virtualenv_module <- function(python) {
  py_version <- python_version(python)

  # prefer 'venv' for Python 3, but allow for 'virtualenv' for both
  # (note that 'venv' and 'virtualenv' are largely compatible)
  modules <- "virtualenv"
  if (py_version >= "3.6")
    modules <- c("venv", modules)

  # if we have one of these modules available, return it
  for (module in modules)
    if (python_has_module(python, module))
      return(module)

  # virtualenv not available: instruct the user to install
  commands <- new_stack()
  commands$push("Tools for managing Python virtual environments are not installed.")
  commands$push("")

  # if we don't have pip, recommend its installation
  if (!python_has_module(python, "pip")) {
    commands$push("Install pip with:")
    if (python_has_module(python, "easy_install")) {
      commands$push(paste("$", python, "-m easy_install --upgrade --user pip"))
    } else if (is_ubuntu() && dirname(python) == "/usr/bin") {
      package <- if (py_version < 3) "python-pip" else "python3-pip"
      commands$push(paste("$ sudo apt-get install", package))
    } else {
      commands$push("$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py")
      commands$push(paste("$", python, "get-pip.py"))
    }
    commands$push("")
  }

  # then, recommend installation of virtualenv or venv with pip
  commands$push(paste("Install", modules[[1]], "with:"))
  if (is_ubuntu() && dirname(python) == "/usr/bin") {
    package <- if (py_version < 3) "python-virtualenv" else "python3-venv"
    commands$push(paste("$ sudo apt-get install", package))
  } else {
    commands$push(paste("$", python, "-m pip install --upgrade --user", module))
  }
  commands$push("\n")

  # report to user
  message <- paste(commands$data(), collapse = "\n")
  stop(message, call. = FALSE)

}



is_virtualenv <- function(dir) {

  # virtual environment created with venv
  if (file.exists(file.path(dir, "pyvenv.cfg")))
    return(TRUE)

  # virtual environment created with virtualenv
  if (file.exists(file.path(dir, "bin/activate_this.py")))
    return(TRUE)

  FALSE
}

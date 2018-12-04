#' Interface to Python Virtual Environments
#'
#' R functions for managing Python [virtual environments](https://virtualenv.pypa.io/en/stable/).
#'
#' @details
#' Virtual environments are by default located at `~/.virtualenvs`. You can change this
#' behavior by defining the `WORKON_HOME` environment variable.
#'
#' Virtual environment functions are not supported on Windows (the use of
#' [conda environments][conda-tools] is recommended on Windows).
#'
#' @param envname The name of, or path to, a Python virtual environment.
#' @param packages A character vector with package names to install or remove.
#' @param ignore_installed Boolean; ignore previously-installed versions of the
#'   requested packages?
#' @param confirm Boolean; confirm before removing packages or virtual
#'   environments?
#' @param python The path to a Python interpreter. When `NULL`, the active
#'   Python interpreter associated with the current session will be used.
#'
#' @name virtualenv-tools
NULL



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_list <- function() {
  root <- virtualenv_root()
  if (file.exists(virtualenv_root()))
    list.files(root)
  else
    character()
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

  # choose appropriate tool for creating virtualenv
  module <- virtualenv_module(python)
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
virtualenv_install <- function(envname, packages, ignore_installed = TRUE) {
  path <- virtualenv_create(envname)

  # ensure we have a recent version of pip installed
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

  # packages = NULL means remove the entire virtualenv
  if (is.null(packages)) {

    if (confirm) {
      prompt <- sprintf("Remove virtualenv at %s? [Y/n]: ", path)
      response <- readline(prompt = prompt)
      if (tolower(response) != "y") {
        writeLines("Operation aborted.")
        return(invisible(NULL))
      }
    }

    unlink(path, recursive = TRUE)
    return(invisible(NULL))

  }

  # otherwise, remove the requested packages
  if (confirm) {
    fmt <- "Remove %s from %s? [Y/n]: "
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

  if (!utils::file_test("-d", path))
    return(FALSE)

  if (!file.exists(file.path(path, "bin/activate")))
    return(FALSE)

  TRUE
}



virtualenv_path <- function(envname) {

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

  config <- py_config()
  normalizePath(config$python, winslash = "/")
}



virtualenv_module <- function(python) {

  py_version <- python_version(python)
  modules <- if (py_version < 3)
    c("virtualenv")
  else
    c("venv", "virtualenv")

  for (module in modules)
    if (python_has_module(python, module))
      return(module)

  # provide some diagnostics for the user
  commands <- new_stack()
  commands$push("tools for interacting with Python virtual environments are not installed.")
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

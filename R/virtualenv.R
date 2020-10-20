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
#' @param pip_options An optional character vector of additional command line
#'   arguments to be passed to `pip`.
#'
#' @param confirm Boolean; confirm before removing packages or virtual
#'   environments?
#'
#' @param python The path to a Python interpreter, to be used with the created
#'   virtual environment. When `NULL`, the Python interpreter associated with
#'   the current session will be used.
#'
#' @param packages A set of Python packages to install (via `pip install`) into
#'   the virtual environment, after it has been created. By default, the
#'   `"numpy"` package will be installed, and the `pip`, `setuptools` and
#'   `wheel` packages will be updated. Set this to `FALSE` to avoid installing
#'   any packages after the virtual environment has been created.
#'
#' @param system_site_packages Boolean; create new virtual environments with
#'   the `--system-site-packages` flag, thereby allowing those virtual
#'   environments to access the system's site packages. Defaults to `FALSE`.
#'
#' @param ... Optional arguments; currently ignored for future expansion.
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
virtualenv_create <- function(
  envname = NULL,
  python = NULL,
  packages = "numpy",
  system_site_packages = getOption("reticulate.virtualenv.system_site_packages", default = FALSE))
{
  path <- virtualenv_path(envname)
  name <- if (is.null(envname)) path else envname

  # check and see if we already have a virtual environment
  if (virtualenv_exists(path)) {
    writeLines(paste("virtualenv:", name))
    return(invisible(path))
  }

  python <- virtualenv_default_python(python)
  module <- virtualenv_module(python)

  # use it to create the virtual environment (note that 'virtualenv'
  # requires us to request the specific Python binary we wish to use when
  # creating the environment)
  args <- c("-m", module)
  if (module == "virtualenv")
    args <- c(args, "-p", shQuote(python))
  
  # add --system-site-packages if requested
  if (system_site_packages)
    args <- c(args, "--system-site-packages")
  
  # add the path where the environment will be created
  args <- c(args, shQuote(path.expand(path)))
  
  writef("Using Python: %s", python)
  printf("Creating virtual environment %s ... ", shQuote(name))
  
  result <- system2(python, args)
  if (result != 0L) {
    writef("FAILED")
    fmt <- "Error creating virtual environment '%s' [error code %d]"
    stopf(fmt, name, result)
  }
  
  writef("Done!")

  # upgrade pip and friends after creating the environment
  # (since the version bundled with virtualenv / venv may be stale)
  if (!identical(packages, FALSE)) {
    python <- virtualenv_python(envname)
    packages <- unique(c("pip", "wheel", "setuptools", packages))
    writef("Installing packages: %s", paste(shQuote(packages), collapse = ", "))
    pip_install(python, packages)
  }
  
  writef("Virtual environment '%s' successfully created.", name)
  invisible(path)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_install <- function(envname = NULL,
                               packages,
                               ignore_installed = FALSE,
                               pip_options = character(),
                               ...)
{
  check_forbidden_install("Python packages")
  
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
  python <- virtualenv_python(envname)
  if (pip_version(python) < "8.1")
    pip_install(python, c("pip", "wheel", "setuptools"))
  
  # now install the requested package
  pip_install(python,
              packages,
              ignore_installed = ignore_installed,
              pip_options = pip_options)
}



#' @inheritParams virtualenv-tools
#' @rdname virtualenv-tools
#' @export
virtualenv_remove <- function(envname = NULL,
                              packages = NULL,
                              confirm = interactive())
{
  path <- virtualenv_path(envname)
  name <- if (is.null(envname)) path else envname
  confirm <- confirm && is_interactive()

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

  python <- virtualenv_python(envname)
  pip_uninstall(python, packages)
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
  suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
  path.expand(file.path(path, suffix))
}



virtualenv_exists <- function(envname = NULL) {

  # try to resolve path
  path <- tryCatch(virtualenv_path(envname), error = identity)
  if (inherits(path, "error"))
    return(FALSE)

  is_virtualenv(path)

}



virtualenv_path <- function(envname = NULL) {

  python_environment_resolve(
    envname = envname,
    resolve = function(envname) file.path(virtualenv_root(), envname)
  )

}


virtualenv_pip <- function(envname) {
  path <- virtualenv_path(envname)
  suffix <- if (is_windows()) "Scripts/pip.exe" else "bin/python"
  path.expand(file.path(path, suffix))
}



virtualenv_default_python <- function(python = NULL) {

  # if the user has supplied a verison of python already, use it
  if (!is.null(python))
    return(path.expand(python))

  # check for some pre-defined Python sources (prefer Python 3)
  pythons <- c(
    Sys.getenv("RETICULATE_PYTHON"),
    .globals$required_python_version,
    Sys.which("python3"),
    Sys.which("python")
  )

  for (python in pythons) {

    # skip non-existent Python
    if (!file.exists(python))
      next

    # get list of required modules
    suppressWarnings({version <- tryCatch(python_version(python), error = identity)})
    if (inherits(version, "error"))
      next

    py2_modules <- c("pip", "virtualenv")
    py3_modules <- c("pip", "venv")
    modules <- ifelse(version < 3, py2_modules, py3_modules)

    # ensure these modules are available
    if (!python_has_modules(python, modules))
      next

    return(normalizePath(python, winslash = "/"))

  }

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
  
  # check for expected files for virtualenv / venv
  subdir <- if (is_windows()) "Scripts" else "bin"
  
  files <- c(
    file.path(subdir, "activate_this.py"),
    file.path(subdir, "pyvenv.cfg"),
    "pyvenv.cfg"
  )
  
  paths <- file.path(dir, files)
  any(file.exists(paths))
  
}

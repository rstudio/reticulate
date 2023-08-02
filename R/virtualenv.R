#' Interface to Python Virtual Environments
#'
#' R functions for managing Python [virtual
#' environments](https://virtualenv.pypa.io/en/stable/).
#'
#' Virtual environments are by default located at `~/.virtualenvs` (accessed
#' with the `virtualenv_root()` function). You can change the default location
#' by defining the `WORKON_HOME` environment variable.
#'
#' Virtual environments are created from another "starter" or "seed" Python
#' already installed on the system. Suitable Pythons installed on the system are
#' found by `virtualenv_starter()`.
#'
#' @param envname The name of, or path to, a Python virtual environment. If this
#'   name contains any slashes, the name will be interpreted as a path; if the
#'   name does not contain slashes, it will be treated as a virtual environment
#'   within `virtualenv_root()`. When `NULL`, the virtual environment as
#'   specified by the `RETICULATE_PYTHON_ENV` environment variable will be used
#'   instead. To refer to a virtual environment in the current working
#'   directory, you can prefix the path with `./<name>`.
#'
#' @param packages A character vector with package names to install or remove.
#'
#' @param requirements Filepath to a pip requirements file.
#'
#' @param ignore_installed Boolean; ignore previously-installed versions of the
#'   requested packages? (This should normally be `TRUE`, so that pre-installed
#'   packages available in the site libraries are ignored and hence packages are
#'   installed into the virtual environment.)
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
#' @param force Boolean; force recreating the environment specified by
#'   `envname`, even if it already exists. If `TRUE`, the previous enviroment is
#'   first deleted and recreated. Otherwise, if `FALSE`, the path to the
#'   existing environment is returned.
#'
#' @param version,python_version The version of Python to use when creating a
#'   virtual environment. Python installations will be searched for using
#'   [`virtualenv_starter()`]. This can a specific version, like `"3.9"` or
#'   `"3.9.3`, or a comma separated list of version constraints, like `">=3.8"`,
#'   or `"<=3.11,!=3.9.3,>3.6"`
#'
#' @param all If `TRUE`, `virtualenv_starter()` returns a 2-column data frame,
#'   with column names `path` and `version`. If `FALSE`, only a single path to a
#'   python binary is returned, corresponding to the first entry when `all =
#'   TRUE`, or `NULL` if no suitable python binaries were found.
#'
#' @param packages A set of Python packages to install (via `pip install`) into
#'   the virtual environment, after it has been created. By default, the
#'   `"numpy"` package will be installed, and the `pip`, `setuptools` and
#'   `wheel` packages will be updated. Set this to `FALSE` to avoid installing
#'   any packages after the virtual environment has been created.
#'
#' @param module The Python module to be used when creating the virtual
#'   environment -- typically, `virtualenv` or `venv`. When `NULL` (the
#'   default), `venv` will be used if available with Python >= 3.6; otherwise,
#'   the `virtualenv` module will be used.
#'
#' @param system_site_packages Boolean; create new virtual environments with the
#'   `--system-site-packages` flag, thereby allowing those virtual environments
#'   to access the system's site packages? Defaults to `FALSE`.
#'
#' @param pip_version The version of `pip` to be installed in the virtual
#'   environment. Relevant only when `module == "virtualenv"`. Set this to
#'   `FALSE` to disable installation of `pip` altogether.
#'
#' @param setuptools_version The version of `setuptools` to be installed in the
#'   virtual environment. Relevant only when `module == "virtualenv"`. Set this
#'   to `FALSE` to disable installation of `setuptools` altogether.
#'
#' @param extra An optional set of extra command line arguments to be passed.
#'   Arguments should be quoted via `shQuote()` when necessary.
#'
#' @param ... Optional arguments; currently ignored and reserved for future
#'   expansion.
#'
#' @name virtualenv-tools
NULL



#' @rdname virtualenv-tools
#' @export
virtualenv_create <- function(
  envname = NULL,
  python  = virtualenv_starter(version),
  ...,
  version              = NULL,
  packages             = "numpy",
  requirements         = NULL,
  force                = FALSE,
  module               = getOption("reticulate.virtualenv.module"),
  system_site_packages = getOption("reticulate.virtualenv.system_site_packages", default = FALSE),
  pip_version          = getOption("reticulate.virtualenv.pip_version", default = NULL),
  setuptools_version   = getOption("reticulate.virtualenv.setuptools_version", default = NULL),
  extra                = getOption("reticulate.virtualenv.extra", default = NULL))
{
  path <- virtualenv_path(envname)
  name <- if (is.null(envname)) path else envname

  # check and see if we already have a virtual environment
  if (virtualenv_exists(path)) {
    if(force) {
      virtualenv_remove(envname = envname, confirm = FALSE)
    } else {
      writeLines(paste("virtualenv:", name))
      return(invisible(path))
    }
  }

  if (is.null(python))
    stop_no_virtualenv_starter(version)

  check_can_be_virtualenv_starter(python)

  module <- module %||% virtualenv_module(python)

  # use it to create the virtual environment
  # (note that 'virtualenv' requires us to request the specific Python binary
  # we wish to use when creating the environment)
  args <- c("-m", module)
  if (module == "virtualenv") {

    # request the specific version of Python
    args <- c(args, "-p", shQuote(python))

    # request specific version of pip
    if (identical(pip_version, FALSE))
      args <- c(args, "--no-pip")
    else if (!is.null(pip_version))
      args <- c(args, "--pip", shQuote(pip_version))

    # request specific version of setuptools
    if (identical(setuptools_version, FALSE))
      args <- c(args, "--no-setuptools")
    else if (!is.null(setuptools_version))
      args <- c(args, "--setuptools", shQuote(setuptools_version))

  }

  # add --system-site-packages if requested
  if (system_site_packages)
    args <- c(args, "--system-site-packages")

  # add in any other arguments provided by the user
  args <- c(args, extra)

  # add the path where the environment will be created
  args <- c(args, maybe_shQuote(path.expand(path)))

  writef("Using Python: %s", python)
  printf("Creating virtual environment %s ... \n", shQuote(name))

  result <- system2t(python, args)
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
    # first upgrade pip and friends
    writef("Installing packages: pip, wheel, setuptools")
    pip_install(python, c("pip", "wheel", "setuptools"))
    packages <- setdiff(packages, c("pip", "wheel", "setuptools"))
    # install requested packages
    if (length(packages)) {
      writef("Installing packages: %s", paste(maybe_shQuote(packages), collapse = ", "))
      pip_install(python, packages)
    }
  }

  if(!is.null(requirements)) {
    writef("Installing packages per requirements file: %s", normalizePath(requirements))
    pip_install(python, requirements = requirements)
  }

  writef("Virtual environment '%s' successfully created.", name)
  invisible(path)
}



#' @rdname virtualenv-tools
#' @export
virtualenv_install <- function(envname = NULL,
                               packages = NULL,
                               ignore_installed = FALSE,
                               pip_options = character(),
                               requirements = NULL,
                               ...,
                               python_version = NULL)
{
  check_forbidden_install("Python packages")

  # check that packages wasn't accidentally supplied to the envname argument
  if (is.null(packages) && is.null(requirements)) {
    if (!is.null(envname)) {

      fmt <- paste(
        "argument \"packages\" is missing, with no default",
        "- did you mean 'virtualenv_install(<envname>, %1$s)'?",
        "- use 'py_install(%1$s)' to install into the active Python environment",
        sep = "\n"
      )

      stopf(fmt, deparse1(substitute(envname)), call. = FALSE)

    }
  }

  # create virtual environment on demand
  path <- virtualenv_path(envname)
  if (!file.exists(path))
    path <- virtualenv_create(envname, version = python_version,
                              packages = NULL)
  # packages=NULL: install only pip, setuptools, wheel, not numpy

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
              pip_options = pip_options,
              requirements = requirements)
}



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

#' @rdname virtualenv-tools
#' @export
virtualenv_list <- function() {
  root <- virtualenv_root()
  if (!file.exists(root))
    return(character())
  list.files(root)
}


#' @rdname virtualenv-tools
#' @export
virtualenv_root <- function() {
  Sys.getenv("WORKON_HOME", unset = "~/.virtualenvs")
}



#' @rdname virtualenv-tools
#' @export
virtualenv_python <- function(envname = NULL) {
  path <- virtualenv_path(envname)
  suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
  path.expand(file.path(path, suffix))
}

#' @rdname virtualenv-tools
#' @export
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


# This function is at this point only invoked from `py_install()`
# no longer safe to call from `virtualenv_create()` due to potential
# for infinite recursion via py_discover_config() bootstrapping a venv.
virtualenv_default_python <- function(python = NULL) {


  # if the user has supplied a version of python already, use it
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

    if(!can_be_virtualenv_starter(python))
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
  commands <- stack(mode = "character")
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



#' @rdname virtualenv-tools
#' @export
virtualenv_starter <- function(version = NULL, all = FALSE) {

  # if `version` is an absolute path to a python binary, reflect it
  if(!is.null(version) &&
     is_string(version) &&
     file.exists(version) &&
     grepl("^python[0-9.]*(\\.exe)?$", basename(version)))
    return(version)

  starters <- data.frame(version = numeric_version(character()),
                         path = character(),
                         stringsAsFactors = FALSE)

  find_starters <- function(glob) {
    # accept NULL, NA, and "" as a no-op
    if(is.null(glob) || isTRUE(is.na(glob)) || isFALSE(nzchar(glob)))
      return(invisible(starters))

    # if not a glob,
    if (!grepl("*", glob, fixed = TRUE)) {
    # accept a path to a directory, convert it into globs that
    # catch two different scenarios:
    # - flat directory of python installations (presumably of different versions)
    # - nested directory of python installations (presumably nested by version and arch)

      suffix <- if (is_windows())
        #e.g, /3.9.4/python.exe, /3.9.4/x64/python.exe
        c("/*/python*.exe", "/*/*/python*.exe")
      else {
        #/3.9.4/bin/python, /3.9.4/x64/bin/python
        c("/*/bin/python*", "/*/*/bin/python*")
      }
      glob <- paste0(normalizePath(glob, winslash = "/", mustWork = FALSE),
                     suffix)
    }

    p <- unique(normalizePath(Sys.glob(glob), winslash = "/"))
    p <- p[grep("^python[0-9.]*(\\.exe)?$", basename(p))]
    v <- numeric_version(vapply(p, function(python_path)
      tryCatch({
        v <- suppressWarnings(system2(
          python_path, "-EV",
          stdout = TRUE, stderr = TRUE))
        # v should be something like "Python 3.10.6"
        if ((attr(v, "status") %||% 0) ||
            length(v) != 1L ||
            !startsWith(v, "Python "))
          return(NA_character_)
        substr(v, 8L, 999L)
      }, error = function(e) NA_character_), ""), strict = FALSE)
    df <- data.frame(version = v, path = p,
                     row.names = NULL, stringsAsFactors = FALSE)
    df <- df[!is.na(df$version), ]
    df <- df[order(df$version, decreasing = TRUE), ]

    df <- rbind(starters, df, stringsAsFactors = FALSE)
    df <- df[!duplicated(df$path), ]
    if(is_windows()) {
      # on windows, removed dups of the same python in the same directory,
      # like 'python.exe', 'python3.exe' 'python3.11.exe'
      df <- df[!duplicated(dirname(df$path)), ]
    }
    rownames(df) <- NULL
    starters <<- df
  }


  for (custom_loc in list(Sys.getenv("RETICULATE_VIRTUALENV_STARTER"),
                          getOption("reticulate.virtualenv.starter", ""))) {
    # Accept user customization, a character vector (or ":" separated string) of
    # file paths to python binaries. Paths can be globs, as they are passed on
    # to Sys.glob()
    lapply(unlist(strsplit(custom_loc, "[:;]")), find_starters)
  }

  # Find pythons installed via `install_python()` or by directly using pyenv.
  # Typically something like "~/.pyenv/versions/3.9.17/bin/python3.9" or
  #  "C:/Users/<username>/AppData/Local/r-reticulate/r-reticulate/pyenv/pyenv-win/versions/3.9.13/python.exe"
  # but can be different if user set PYENV_ROOT or manually installed pyenv
  # in a different location
  if (!is.null(pyenv <- pyenv_find(install = FALSE))) {
     if (is_windows()) {
       pyenv_root <- dirname(dirname(pyenv))
       find_starters(file.path(pyenv_root, "versions/*/python*.exe"))
     } else {
       pyenv_root <- system2(pyenv, "root", stdout = TRUE)
       find_starters(file.path(pyenv_root, "versions/*/bin/python*"))
    }
  }

  # official python.org installer for macOS default location
  # "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
  if (is_macos())
    find_starters("/Library/Frameworks/Python.framework/Versions/*/bin/python*")

  # official python.org installer for windows
  # system install:  "C:/Program Files/Python311/python.exe"
  # user install: "C:/Users/<username>/AppData/Local/Programs/Python/Python311/python.exe"
  # TODO: we can make this more robust by using env vars SYSTEMDRIVE, and USERPROFILE
  # see https://github.com/rstudio/rstudio/blob/094af5c40cd13ef8ac84845462c35ffeb3a06d65/src/cpp/session/modules/SessionPythonEnvironments.R#L555C28-L555C39
  if (is_windows()) {
    find_starters("/Program Files/Python*/python*.exe")
    find_starters("~/../AppData/Local/Programs/Python/Python*/python*.exe")
  }

  # Pythons installed from https://github.com/rstudio/python-builds
  # e.g., "/opt/python/3.11.4/bin/python3.11"
  if (is_linux())
    find_starters("/opt/python/*/bin/python*")

  # python installed system wide
  if (!is_windows())
    find_starters("/usr/local/bin/python*")

  # only use system python on linux, not mac
  if (is_linux())
    find_starters("/usr/bin/python*")

  # on macOS, intentionally don't discover homebrew python
  # https://justinmayer.com/posts/homebrew-python-is-not-for-you/
  # if (is_macos()) find_starters("/opt/homebrew/opt/python*/bin/python*")

  # on Github Action Runners, find Pythons installed in the tool cache
  if(!is.na(tool_cache_dir <- Sys.getenv("RUNNER_TOOL_CACHE", NA)))
    find_starters(paste0(tool_cache_dir, "/Python"))

  # if specific version requested, filter for that.
  if (!is.null(version)) {
    for (check in as_version_constraint_checkers(version)) {
      satisfies_constraint <- check(starters$version)
      starters <- starters[satisfies_constraint, ]
    }
    rownames(starters) <- NULL
  }

  if (all)
    starters
  else if (nrow(starters))
    starters$path[[1L]]
  else
    NULL

}

as_version_constraint_checkers <- function(version) {

  if (inherits(version, "numeric_version"))
    version <- as.character(version)
  stopifnot(is.character(version))

  # given a version string like ">=3.6,!=3.9,<3.11", split on ","
  version <- unlist(strsplit(version, ",", fixed = TRUE))

  # given string like ">=3.8", match two groups, on ">=" and "3.8"
  pattern <- "^([><=!]{0,2})\\s*([0-9.]*)"

  op <- sub(pattern, "\\1",  version)
  op[op == ""] <- "=="

  ver <- sub(pattern, "\\2",  version)
  ver <- numeric_version(ver)

  .mapply(function(op, ver) {
    op <- get(op, mode = "function")
    force(ver)

    # return a "checker" function that takes a vector of versions and returns
    # a logical vector of if the version satisfies the constraint.
    function(x) {
      x <- numeric_version(x)
      # if the constraint version is missing minor or patch level, set
      # to 0, so we can match on all, equivalent to pip style syntax like '3.8.*'
      for (lvl in 3:2)
        if (is.na(ver[, lvl])) {
          ver[, lvl] <- 0L
          x[, lvl] <- 0L
        }
      op(x, ver)
    }
  }, list(op, ver), NULL)
}


check_can_be_virtualenv_starter <- function(python) {
  if(!can_be_virtualenv_starter(python))
    stop_no_virtualenv_starter()
}

can_be_virtualenv_starter <- function(python) {
  if (!file.exists(python))
    return(FALSE)

  # get version
  version <- tryCatch(
    suppressWarnings(python_version(python)),
    error = identity
  )

  if (inherits(version, "error"))
    return(FALSE)

  py2_modules <- c("pip", "virtualenv")
  py3_modules <- c("pip", "venv")
  modules <- ifelse(version < 3, py2_modules, py3_modules)

  # ensure these modules are available
  if (!python_has_modules(python, modules))
    return(FALSE)

  # check if python was built with `--enable-shared`, to make sure
  # we don't bootstrap a venv that reticulate can't bind to
  config <- python_config(python, modules)
  if (is.null(config$libpython))
    return(FALSE)

  TRUE
}


stop_no_virtualenv_starter <- function(version = NULL) {

  .msg <- character()
  w <- function(...) .msg <<- c(.msg, paste0(...))

  w("Suitable Python installation for creating a venv not found.")
  if (!is.null(version))
    w("Requested version constraint: ", version)
  w("Please install Python with one of following methods:")

  if (is_linux())
      w("- https://github.com/rstudio/python-builds/")

  if (!is_linux())
    w("- https://www.python.org/downloads/")

  w("- reticulate::install_python(version = '<version>')")

  stop(paste0(.msg, collapse = "\n"))

}

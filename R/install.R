
py_install_method_detect <- function(envname, conda = "auto") {

  # try to find an existing virtualenv
  if (virtualenv_exists(envname))
    return("virtualenv")

  # try to find an existing condaenv
  if (condaenv_exists(envname, conda = conda))
    return("conda")

  # check to see if virtualenv or venv is available
  python <- virtualenv_starter()
  if (!is.null(python) &&
      (python_has_module(python, "venv") ||
       python_has_module(python, "virtualenv")))
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

  if (is.null(envname) && is_ephemeral_venv_initialized()) {
    if (!is.null(python_version)) {
      stop(
        "Python version requirements cannot be ",
        "changed after Python has been initialized"
      )
    }
    warning(
      "An ephemeral virtual environment managed by 'reticulate' is currently in use.\n",
      "To add more packages to your current session, call `py_require()` instead\n",
      "of `py_install()`. Running:\n  ",
      paste0(
        "`py_require(c(", paste0(sprintf("\"%s\"", packages), collapse = ", "), "))`"
      )
    )
    py_require(packages)
    return(invisible())
  }

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
      python_version = python_version,
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


#' Write and read Python requirements files
#'
#' @description
#'
#' - `py_write_requirements()` writes the requirements currently tracked by
#' [py_require()]. If `freeze = TRUE` or if the `python` environment is not
#' ephemeral, it writes a fully resolved manifest via `pip freeze`.
#'
#' - `py_read_requirements()` reads `requirements.txt` and `.python-version`, and
#' applies them with [py_require()]. By default, entries are added (`action
#' = "add"`).
#'
#' These are primarily an alternative interface to `py_require()`, but can
#' also work with non-ephemeral virtual environments.
#'
#' @note
#'
#' To continue using `py_require()` locally while keeping a
#' `requirements.txt` up-to-date for deployments, you can register
#' an exit handler in `.Rprofile` like this:
#'
#' ```r
#' reg.finalizer(
#'   asNamespace("reticulate"),
#'   function(ns) {
#'     if (
#'       reticulate::py_available() &&
#'         isTRUE(reticulate::py_config()$ephemeral)
#'     ) {
#'       reticulate::py_write_requirements(quiet = TRUE)
#'     }
#'   },
#'   onexit = TRUE
#' )
#' ```
#'
#' This approach is only recommended if you are using `git`.
#'
#' Alternatively, you can transition away from using ephemeral python
#' environemnts via `py_require()` to using a persistent local virtual
#' environment you manage. You can create a local virtual environment from
#' `requirements.txt` and `.python-version` using [virtualenv_create()]:
#'
#' ```r
#' # Note: '.venv' in the current directory is auto-discovered by reticulate.
#' # https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery
#' virtualenv_create(
#'   "./.venv",
#'   version = readLines(".python-version"),
#'   requirements = "requirements.txt"
#' )
#' ```
#'
#' If you run into issues, be aware that `requirements.txt` and
#' `.python-version` may not contain all the information necessary to
#' reproduce the Python environment if the R code sets environment variables
#' like `UV_INDEX` or `UV_CONSTRAINT`.
#'
#' @name py_requirements_files
#'
#' @param packages Path to the package requirements file. Defaults to
#'   `"requirements.txt"`. Use `NULL` to skip.
#' @param python_version Path to the Python version file. Defaults to
#'   `".python-version"`. Use `NULL` to skip.
#' @param freeze Logical. If `TRUE`, writes a fully resolved list of
#'   installed packages using `pip freeze`. If `FALSE`, writes only the
#'   requirements tracked by [py_require()].
#' @param python Path to the Python executable to use.
#' @param action How to apply requirements read by `py_read_requirements()`:
#'   `"add"` (default) adds to existing requirements, `"set"` replaces them,
#'   `"remove"` removes matching entries, or `"none"` skips applying them
#'   and returns the read values.
#' @param ... Unused; must be empty.
#' @param quiet Logical; if `TRUE`, suppresses the informational messages
#'   that print `wrote '<path>'` for each file written.
#'
#' @return Invisibly, a list with two named elements:
#' \describe{
#'   \item{`packages`}{Character vector of package requirements.}
#'   \item{`python_version`}{String specifying the Python version.}
#' }
#'
#'   To get just the return value without writing any files, you can pass
#'   `NULL` for file paths, like this:
#'
#' ```r
#' py_write_requirements(NULL, NULL)
#' py_write_requirements(NULL, NULL, freeze = TRUE)
#' ```
#'
#' @export
py_write_requirements <- function(
  packages = "requirements.txt",
  python_version = ".python-version",
  ...,
  freeze = NULL,
  python = py_exe(),
  quiet = FALSE
) {
  rlang::check_dots_empty()
  if (
    (is.null(python) || is_ephemeral_venv_initialized(python)) &&
    (is.null(freeze) || isFALSE(freeze))
  ) {
    reqs <- py_require()

    pkgs <- reqs$packages
    ver <- reqs$python_version
    ver <- resolve_python_version(ver)
    if (is.null(reqs$python_version)) {
      ver <- sub("\\.[0-9]+$", "", ver)
    }
  } else {
    if (isFALSE(freeze)) {
      stop(
        "freeze = FALSE is only supported for environments assembled via py_require()."
      )
    }
    if (is.null(python)) {
      python <- uv_get_or_create_env()
    }

    pkgs <- uv_exec(
      c("pip freeze --no-progress --color never --python",
        maybe_shQuote(python)),
      stdout = TRUE, stderr = FALSE
    )
    ver <- system2(python, c("-E -V"), stdout = TRUE)
    ver <- sub("^Python\\s+", "", ver)
  }

  if (!is.null(packages)) {
    writeLines(pkgs, packages)
    if (!quiet) message("Wrote '", packages, "'")
  }

  if (!is.null(python_version)) {
    writeLines(ver, python_version)
    if (!quiet) message("Wrote '", python_version, "'")
  }

  out <- list(packages = pkgs, python_version = ver)
  if (is.null(packages) && is.null(python_version)) out else invisible(out)
}

#' @rdname py_requirements_files
#' @export
py_read_requirements <- function(
  packages = "requirements.txt",
  python_version = ".python-version",
  ...,
  action = c("add", "set", "remove", "none")
) {
  rlang::check_dots_empty()
  action <- match.arg(action)

  read_requirements <- function(path, stop_if_missing = TRUE) {
    if (is.null(path)) return(NULL)
    if (!file.exists(path)) {
      if (as.logical(stop_if_missing)) {
        stop(
          "File '", path, "' does not exist in the current working directory.",
        )
      } else {
        return(NULL)
      }
    }
    lines <- readLines(path, warn = FALSE)
    lines <- trimws(lines)
    lines <- lines[nzchar(lines)]
    lines <- lines[!startsWith(lines, "#")]
    lines
  }

  pkgs <- read_requirements(packages, stop_if_missing = length(packages))
  py_ver <- read_requirements(python_version, stop_if_missing = FALSE)
  if (!length(py_ver)) py_ver <- NULL

  out <- list(packages = pkgs, python_version = py_ver)
  if (action == "none") {
    out
  } else {
    py_require(pkgs, py_ver, action = action)
    invisible(out)
  }
}

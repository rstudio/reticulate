
#' @param path The location where Miniconda is (or should be) installed. Note
#'   that the Miniconda installer does not support paths containing spaces. See
#'   [miniconda_path] for more details on the default path used by `reticulate`.
#'
#' @title miniconda-params
#' @keywords internal
#' @name miniconda-params
NULL

#' Install Miniconda
#'
#' Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
#' installer, and use it to install Miniconda.
#'
#' For arm64 builds of R on macOS, `install_miniconda()` will use
#' binaries from [miniforge](https://github.com/conda-forge/miniforge) instead.
#'
#' @inheritParams miniconda-params
#'
#' @param update Boolean; update to the latest version of Miniconda after
#'   installation?
#'
#' @param force Boolean; force re-installation if Miniconda is already installed
#'   at the requested path?
#'
#' @note If you encounter binary incompatibilities between R and Miniconda, a
#'   scripted build and installation of Python from sources can be performed by
#'   [`install_python()`]
#'
#' @family miniconda-tools
#' @export
install_miniconda <- function(path = miniconda_path(),
                              update = TRUE,
                              force = FALSE)
{
  check_forbidden_install("Miniconda")

  path <- path.expand(path)

  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install Miniconda into a path containing spaces")

  # TODO: what behavior when miniconda is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniconda_preflight(path, force)

  # download the installer
  message("* Installing Miniconda -- please wait a moment ...")
  url <- miniconda_installer_url()
  installer <- miniconda_installer_download(url)

  if (force) {
    # miniconda installer '-u' (update) flag frequently does not work, errors.
    miniconda_uninstall(path)
  }

  # run the installer
  miniconda_installer_run(installer, update, path)

  # validate the install succeeded
  ok <- miniconda_exists(path) && miniconda_test(path)
  if (!ok)
    stopf("Miniconda installation failed [unknown reason]")

  # update to latest version if requested
  if (update)
    miniconda_update(path)

  # create r-reticulate environment
  conda <- miniconda_conda(path)
  python <- miniconda_python_package()
  conda_create("r-reticulate", packages = c(python, "numpy"), conda = conda)

  messagef("* Miniconda has been successfully installed at %s.", pretty_path(path))
  path

}

#' Update Miniconda
#'
#' Update Miniconda to the latest version.
#'
#' @inheritParams miniconda-params
#'
#' @family miniconda-tools
#' @export
miniconda_update <- function(path = miniconda_path()) {
  conda <- miniconda_conda(path)
  local_conda_paths(conda)
  system2t(conda, c("update", "--yes", "--name", "base", "conda"))
}

#' Remove Miniconda
#'
#' Uninstall Miniconda.
#'
#' @param path The path in which Miniconda is installed.
#'
#' @family miniconda-tools
#' @export
miniconda_uninstall <- function(path = miniconda_path()) {
  unlink(path, recursive = TRUE)
}

install_miniconda_preflight <- function(path, force) {

  # if we're forcing installation, then proceed
  if (force)
    return(invisible(TRUE))

  # if the directory doesn't exist, that's fine
  if (!file.exists(path))
    return(invisible(TRUE))

  # check for a miniconda installation
  if (miniconda_exists(path)) {

    fmt <- paste(
      "Miniconda is already installed at path %s.",
      "- Use `reticulate::install_miniconda(force = TRUE)` to overwrite the previous installation.",
      sep = "\n"
    )

    stopf(fmt, pretty_path(path))
  }

  if (length(list.files(path))) {
    fmt <- paste(
      "Directory %s is not empty.",
      "- Existing files will be permanently deleted during installation.",
      sep = "\n"
    )

    stopf(fmt, pretty_path(path))
  }

  # ok to proceed
  invisible(TRUE)

}

miniconda_installer_url <- function(version = "3") {

  url <- getOption("reticulate.miniconda.url")
  if (!is.null(url))
    return(url)

  base <- "https://github.com/conda-forge/miniforge/releases/latest/download"

  info <- as.list(Sys.info())
  arch <- miniconda_installer_arch(info)
  version <- as.character(version)
  name <- if (is_windows())
    sprintf("Miniforge%s-Windows-%s.exe", version, arch)
  else if (is_osx())
    sprintf("Miniforge%s-MacOSX-%s.sh", version, arch)
  else if (is_linux())
    sprintf("Miniforge%s-Linux-%s.sh", version, arch)
  else
    stopf("unsupported platform %s", shQuote(Sys.info()[["sysname"]]))

  file.path(base, name)

}

miniconda_installer_arch <- function(info) {

  # allow user override
  arch <- getOption("reticulate.miniconda.arch")
  if (!is.null(arch))
    return(arch)

  # miniconda url use x86_64 not x86-64 for Windows
  if (info$machine == "x86-64")
    return("x86_64")

  # otherwise, use arch as-is
  info$machine

}

miniconda_installer_download <- function(url) {

  # reuse an already-existing installer
  installer <- file.path(tempdir(), basename(url))
  if (file.exists(installer))
    return(installer)

  # doesn't exist; try to download it
  messagef("* Downloading %s ...", shQuote(url))
  status <- download.file(url, destfile = installer, mode = "wb")
  if (!file.exists(installer)) {
    fmt <- "download of Miniconda installer failed [status = %i]"
    stopf(fmt, status)
  }

  # download successful; provide file path
  installer

}


miniconda_installer_run <- function(installer, update, path) {

  args <- if (is_windows()) {
    if(dir.exists(path))
      unlink(path, recursive = TRUE)
    c(
      "/InstallationType=JustMe",
      "/AddToPath=0",
      "/RegisterPython=0",
      "/NoRegistry=1",
      "/S",
      paste("/D", utils::shortPathName(path), sep = "=")
    )

  } else if (is_unix()) {
    c("-b", if (update) "-u", "-p", shQuote(path))
  } else {
    stopf("unsupported platform %s", shQuote(Sys.info()[["sysname"]]))
  }

  Sys.chmod(installer, mode = "0755")

  # work around rpath issues on macOS
  #
  # dyld: Library not loaded: @rpath/libz.1.dylib
  # Referenced from: /Users/kevinushey/Library/r-miniconda/conda.exe
  #   Reason: image not found
  #
  # https://github.com/rstudio/reticulate/issues/874
  if (is_osx()) {

    old <- Sys.getenv("DYLD_FALLBACK_LIBRARY_PATH")
    new <- if (nzchar(old))
      paste(old, "/usr/lib", sep = ":")
    else
      "/usr/lib"

    Sys.setenv(DYLD_FALLBACK_LIBRARY_PATH = new)
    on.exit(Sys.setenv(DYLD_FALLBACK_LIBRARY_PATH = old), add = TRUE)

  }
  if (is_windows()) {
    installer <- normalizePath(installer)
    status <- system2t(installer, args)
  }
  if (is_unix()) {
    ##check for bash
    bash_path <- Sys.which("bash")
    if (bash_path[1] == "")
      stopf("The miniconda installer requires bash.")
    status <- system2t(bash_path[1], c(installer, args))
  }
  if (status != 0)
    stopf("miniconda installation failed [exit code %i]", status)

  invisible(path)

}

#' Path to Miniconda
#'
#' The path to the Miniconda installation to use. By default, an OS-specific
#' path is used. If you'd like to instead set your own path, you can set the
#' `RETICULATE_MINICONDA_PATH` environment variable.
#'
#' @family miniconda
#'
#' @export
miniconda_path <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PATH", unset = miniconda_path_default())
}

miniconda_path_default <- function() {

  if (is_osx()) {

    # on macOS, use different path for arm64 miniconda
    path <- if (Sys.info()[["machine"]] == "arm64")
      "~/Library/r-miniconda-arm64"
    else
      "~/Library/r-miniconda"

    return(path.expand(path))

  }

  # otherwise, use rappdirs default
  root <- normalizePath(user_data_dir(), winslash = "/", mustWork = FALSE)
  file.path(root, "r-miniconda")

}

miniconda_exists <- function(path = miniconda_path()) {
  conda <- miniconda_conda(path)
  file.exists(conda)
}

miniconda_test <- function(path = miniconda_path()) {
  python <- python_binary_path(path)
  status <- tryCatch(python_version(python), error = identity)
  !inherits(status, "error")
}

miniconda_conda <- function(path = miniconda_path()) {
  exe <- if (is_windows()) "condabin/conda.bat" else "bin/conda"
  file.path(path, exe)
}

miniconda_envpath <- function(env = NULL, path = miniconda_path()) {
  env <- env %||% Sys.getenv("RETICULATE_MINICONDA_ENVNAME", unset = "r-reticulate")

  if(env == 'base')
    return(path)

  file.path(path, "envs", env)
}

# the default environment path to use for miniconda
miniconda_python_envpath <- function() {

  Sys.getenv(
    "RETICULATE_MINICONDA_PYTHON_ENVPATH",
    unset = miniconda_envpath()
  )

}

# the version of python to use in the environment
miniconda_python_version <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PYTHON_VERSION", unset = "3.12")
}

miniconda_python_package <- function() {
  paste("python", miniconda_python_version(), sep = "=")
}

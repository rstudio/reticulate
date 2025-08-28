
#' @param path The location where Miniforge is (or should be) installed. Note
#'   that the Miniforge installer does not support paths containing spaces. See
#'   [miniforge_path] for more details on the default path used by `reticulate`.
#'
#' @title miniforge-params
#' @keywords internal
#' @name miniforge-params
NULL

#' Install Miniforge
#'
#' Download the [Miniforge](https://github.com/conda-forge/miniforge)
#' installer, and use it to install Miniforge.
#'
#' @inheritParams miniforge-params
#'
#' @param update Boolean; update to the latest version of Miniforge after
#'   installation?
#'
#' @param force Boolean; force re-installation if Miniforge is already installed
#'   at the requested path?
#'
#' @note If you encounter binary incompatibilities between R and Miniforge, a
#'   scripted build and installation of Python from sources can be performed by
#'   [`install_python()`]
#'
#' @family miniforge-tools
#' @export
install_miniforge <- function(path = miniforge_path(),
                              update = TRUE,
                              force = FALSE)
{
  check_forbidden_install("Miniforge")

  path <- path.expand(path)

  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install Miniforge into a path containing spaces")

  # TODO: what behavior when miniforge is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniforge_preflight(path, force)

  # download the installer
  message("* Installing Miniforge -- please wait a moment ...")
  url <- miniforge_installer_url()
  installer <- miniforge_installer_download(url)

  if (force) {
    # miniforge installer '-u' (update) flag frequently does not work, errors.
    miniforge_uninstall(path)
  }

  # run the installer
  miniforge_installer_run(installer, update, path)

  # validate the install succeeded
  ok <- miniforge_exists(path) && miniforge_test(path)
  if (!ok)
    stopf("Miniforge installation failed [unknown reason]")

  # update to latest version if requested
  if (update)
    miniforge_update(path)

  # create r-reticulate environment
  conda <- miniforge_conda(path)
  python <- miniforge_python_package()
  conda_create("r-reticulate", packages = c(python, "numpy"), conda = conda)

  messagef("* Miniforge has been successfully installed at %s.", pretty_path(path))
  path

}

#' Update Miniforge
#'
#' Update Miniforge to the latest version.
#'
#' @inheritParams miniforge-params
#'
#' @family miniforge-tools
#' @export
miniforge_update <- function(path = miniforge_path()) {
  conda <- miniforge_conda(path)
  local_conda_paths(conda)
  system2t(conda, c("update", "--yes", "--name", "base", "conda"))
}

#' Remove Miniforge
#'
#' Uninstall Miniforge.
#'
#' @param path The path in which Miniforge is installed.
#'
#' @family miniforge-tools
#' @export
miniforge_uninstall <- function(path = miniforge_path()) {
  unlink(path, recursive = TRUE)
}

install_miniforge_preflight <- function(path, force) {

  # if we're forcing installation, then proceed
  if (force)
    return(invisible(TRUE))

  # if the directory doesn't exist, that's fine
  if (!file.exists(path))
    return(invisible(TRUE))

  # check for a miniforge installation
  if (miniforge_exists(path)) {

    fmt <- paste(
      "Miniforge is already installed at path %s.",
      "- Use `reticulate::install_miniforge(force = TRUE)` to overwrite the previous installation.",
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

miniforge_installer_url <- function(version = "3") {

  url <- getOption("reticulate.miniforge.url")
  if (!is.null(url))
    return(url)

  base <- "https://github.com/conda-forge/miniforge/releases/latest/download"

  info <- as.list(Sys.info())
  arch <- miniforge_installer_arch(info)
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

miniforge_installer_arch <- function(info) {

  # allow user override
  arch <- getOption("reticulate.miniforge.arch")
  if (!is.null(arch))
    return(arch)

  # miniforge url use x86_64 not x86-64 for Windows
  if (info$machine == "x86-64")
    return("x86_64")

  # otherwise, use arch as-is
  info$machine

}

miniforge_installer_download <- function(url) {

  # reuse an already-existing installer
  installer <- file.path(tempdir(), basename(url))
  if (file.exists(installer))
    return(installer)

  # doesn't exist; try to download it
  messagef("* Downloading %s ...", shQuote(url))
  status <- download.file(url, destfile = installer, mode = "wb")
  if (!file.exists(installer)) {
    fmt <- "download of Miniforge installer failed [status = %i]"
    stopf(fmt, status)
  }

  # download successful; provide file path
  installer

}


miniforge_installer_run <- function(installer, update, path) {

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
  # Referenced from: /Users/kevinushey/Library/r-miniforge/conda.exe
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
      stopf("The miniforge installer requires bash.")
    status <- system2t(bash_path[1], c(installer, args))
  }
  if (status != 0)
    stopf("miniforge installation failed [exit code %i]", status)

  invisible(path)

}

#' Path to Miniforge
#'
#' The path to the Miniforge installation to use. By default, an OS-specific
#' path is used. If you'd like to instead set your own path, you can set the
#' `RETICULATE_MINIFORGE_PATH` environment variable.
#'
#' @family miniforge
#'
#' @export
miniforge_path <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PATH", unset = miniforge_path_default())
}

miniforge_path_default <- function() {

  if (is_osx()) {

    # on macOS, use different path for arm64 miniforge
    path <- if (Sys.info()[["machine"]] == "arm64")
      "~/Library/r-miniforge-arm64"
    else
      "~/Library/r-miniforge"

    return(path.expand(path))

  }

  # otherwise, use rappdirs default
  root <- normalizePath(user_data_dir(), winslash = "/", mustWork = FALSE)
  file.path(root, "r-miniforge")

}

miniforge_exists <- function(path = miniforge_path()) {
  conda <- miniforge_conda(path)
  file.exists(conda)
}

miniforge_test <- function(path = miniforge_path()) {
  python <- python_binary_path(path)
  status <- tryCatch(python_version(python), error = identity)
  !inherits(status, "error")
}

miniforge_conda <- function(path = miniforge_path()) {
  exe <- if (is_windows()) "condabin/conda.bat" else "bin/conda"
  file.path(path, exe)
}

miniforge_envpath <- function(env = NULL, path = miniforge_path()) {
  env <- env %||% Sys.getenv("RETICULATE_MINICONDA_ENVNAME", unset = "r-reticulate")

  if(env == 'base')
    return(path)

  file.path(path, "envs", env)
}

# the default environment path to use for miniforge
miniforge_python_envpath <- function() {

  Sys.getenv(
    "RETICULATE_MINICONDA_PYTHON_ENVPATH",
    unset = miniforge_envpath()
  )

}

# the version of python to use in the environment
miniforge_python_version <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PYTHON_VERSION", unset = "3.10")
}

miniforge_python_package <- function() {
  paste("python", miniforge_python_version(), sep = "=")
}

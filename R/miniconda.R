
#' Install Miniconda
#'
#' Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
#' installer, and use it to install Miniconda.
#'
#' @param path The path in which Miniconda will be installed. Note that the
#'   installer does not support paths containing spaces.
#' 
#' @param version The version of the Miniconda installer to be used. Must be
#'   either 3 (for a Python 3.x installation) or 2 (for a Python 2.x
#'   installation).
#' 
#' @param update Boolean; update to the latest version of Miniconda after install?
#' 
#' @param python The version of Python to use. You can request specific
#'   versions of Python -- for example, to install Python 3.6, you can use
#'   `"python=3.6"`. By default, the latest version of Python is used.
#'   When `"python"` (the default), the latest version of Python as supported by
#'   Miniconda will be used.
#' 
#' @param packages A vector of Python packages to install into the Miniconda
#'   installation. By default, `numpy` is installed into the `r-reticulate`
#'   environment.
#'   
#' @param force Boolean; force re-installation if Miniconda is already installed
#'   at the requested path?
#' 
#' @family miniconda
#' 
#' @export
install_miniconda <- function(
    path = miniconda_path(),
    version = 3,
    update = TRUE,
    python = "python",
    packages = c("numpy"),
    force = FALSE)
{
  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install Miniconda into a path containing spaces")
  
  # NOTE: we'll allow 4 for a future potential Python 4.x release
  if (!version %in% c(2, 3, 4))
    stopf("no known miniconda installer for version %i", as.integer(version))
  
  # TODO: what behavior when miniconda is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniconda_preflight(path, force)
  
  # download the installer
  url <- miniconda_installer_url(version)
  installer <- file.path(tempdir(), basename(url))
  messagef("* Downloading %s ...", shQuote(url))
  status <- download.file(url, destfile = installer, mode = "wb")
  if (!file.exists(installer)) {
    fmt <- "download of Miniconda installer failed [status = %i]"
    stopf(fmt, status)
  }
  
  # run the installer
  message("* Installing Miniconda -- please wait a moment ...")
  miniconda_installer_run(installer, path)
  
  # validate the install succeeded
  ok <- miniconda_exists(path) && miniconda_test(path)
  if (!ok)
    stopf("Miniconda installation failed [unknown reason]")
  
  # update to latest version if requested
  if (update)
    miniconda_update(path)
  
  # create r-reticulate environment
  conda <- miniconda_conda(path)
  packages <- unique(c(python, packages))
  conda_create("r-reticulate", packages = packages, conda = conda)
  
  messagef("* Miniconda has been successfully installed at %s.", shQuote(path))
  path
  
}

#' Update Miniconda
#' 
#' Update Miniconda to the latest version. 
#' 
#' @param path The path in which Miniconda will be installed. Note that the
#'   installer does not support paths containing spaces.
#'
#' @family miniconda
#' 
#' @export
miniconda_update <- function(path = miniconda_path()) {
  conda <- miniconda_conda(path)
  system2(conda, c("update", "--yes", "--name", "base", "conda"))
}

install_miniconda_preflight <- function(path, force) {
  
  # if we're forcing installation, then proceed
  if (force)
    return(invisible(TRUE))
  
  # if the directory doesn't exist, that's fine
  if (!file.exists(path))
    return(invisible(TRUE))
  
  # check for a miniconda installation
  if (miniconda_exists(path))
    stopf("Miniconda is already installed at %s", shQuote(path))
  
  # ok to proceed
  invisible(TRUE)
  
}

miniconda_installer_url <- function(version = "3") {
  
  base <- "https://repo.anaconda.com/miniconda"
  arch <- ifelse(.Machine$sizeof.pointer == 8, "x86_64", "x86")
  version <- as.character(version)
  name <- if (is_windows())
    sprintf("Miniconda%s-latest-Windows-%s.exe", version, arch)
  else if (is_osx())
    sprintf("Miniconda%s-latest-MacOSX-%s.sh", version, arch)
  else if (is_linux())
    sprintf("Miniconda%s-latest-Linux-%s.sh", version, arch)
  else
    stopf("unsupported platform %s", shQuote(Sys.info()[["sysname"]]))
  
  file.path(base, name)
  
}

miniconda_installer_run <- function(installer, path) {
  
  args <- if (is_windows()) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    c(
      "/InstallationType=JustMe",
      "/RegisterPython=0",
      "/S",
      paste("/D", utils::shortPathName(path), sep = "=")
    )
    
  } else if (is_unix()) {
    
    c("-b", "-p", shQuote(path))
    
  } else {
    stopf("unsupported platform %s", shQuote(Sys.info()[["sysname"]]))
  }
  
  Sys.chmod(installer, mode = "0755")
  status <- system2(installer, args)
  if (status != 0)
    stopf("miniconda installation failed [exit code %i]", status)
  
  invisible(path)
  
}

#' Path to Miniconda
#' 
#' The path to the Miniconda installation to use.
#' 
#' @family miniconda
#' 
#' @export
miniconda_path <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PATH", unset = miniconda_path_default())
}

miniconda_path_default <- function() {
  
  if (is_osx())
    return(path.expand("~/Library/r-miniconda"))
  
  root <- normalizePath(rappdirs::user_data_dir(), winslash = "/", mustWork = FALSE)
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
  conda <- file.path(path, exe)
  conda_binary(conda)
}

miniconda_envpath <- function(env, path = miniconda_path()) {
  file.path(path, "envs", env)
}

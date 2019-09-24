
#' Install Miniconda
#'
#' Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
#' installer, and use it to install Miniconda.
#'
#' @param path The path in which Miniconda will be installed. Note that the
#'   installer does not support paths containing spaces.
#' 
#' @param version The (major) version of Python to use.
#' 
#' @param update Boolean; update to the latest version of Miniconda after install?
#' 
#' @param packages A vector of Python packages to install into the Miniconda
#'   installation.
#'   
#' @param force Boolean; force re-installation if Miniconda is already installed
#'   at the requested path?
#' 
#' @family miniconda
#' 
#' @export
install_miniconda <- function(
  path = miniconda_path(),
  version = Sys.getenv("RETICULATE_MINICONDA_PYTHON_VERSION", unset = "3"),
  update = TRUE,
  packages = c("numpy", "scipy", "pandas"),
  force = FALSE)
{
  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install miniconda into a path containing spaces")
  
  # TODO: what behavior when miniconda is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniconda_preflight(path, force)
  
  # download the installer
  url <- miniconda_installer_url(version)
  installer <- file.path(tempdir(), basename(url))
  messagef("* Downloading %s ...", shQuote(url))
  status <- download.file(url, destfile = installer, mode = "wb")
  if (!file.exists(installer)) {
    fmt <- "download of miniconda installer failed [status = %i]"
    stopf(fmt, status)
  }
  
  # run the installer
  message("* Installing miniconda ...")
  miniconda_installer_run(installer, path)
  
  # validate the install succeeded
  if (!miniconda_exists(path))
    stopf("miniconda installation failed [unknown reason]")
  
  # update to latest version
  if (update)
    miniconda_update(path)
  
  # create r-reticulate environment
  conda <- miniconda_conda(path)
  conda_create("r-reticulate", packages = c("python", "numpy"), conda = conda)
  
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
    stopf("miniconda is already installed at %s", shQuote(path))
  
  # ok to proceed
  invisible(TRUE)
  
}

miniconda_installer_url <- function(version = "3") {
  
  base <- "https://repo.anaconda.com/miniconda"
  arch <- ifelse(.Machine$sizeof.pointer == 8, "x86_64", "x86")
  name <- if (is_windows())
    sprintf("Miniconda%s-latest-Windows-%s.exe", version, arch)
  else if (is_osx())
    sprintf("Miniconda%s-latest-MacOSX-%s.sh", version, arch)
  else if (is_unix())
    sprintf("Miniconda%s-latest-Linux-%s.sh", version, arch)
  else
    stopf("unsupported platform %s", shQuote(Sys.info()[["sysname"]]))
  
  file.path(base, name)
  
}

miniconda_installer_run <- function(installer, path) {
  
  args <- if (is_windows()) {
    
    c(
      "/InstallationType=JustMe",
      "/RegisterPython=0",
      "/S",
      shQuote(paste("/D", path, sep = "="))
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
    path.expand("~/Library/r-miniconda")
  else
    rappdirs::user_data_dir("r-miniconda")
}

miniconda_exists <- function(path = miniconda_path()) {
  
  # check for license file
  license <- file.path(path, "LICENSE.txt")
  if (!file.exists(license))
    return(FALSE)
  
  contents <- readLines(license, warn = FALSE)
  any(grepl("Miniconda", contents))
  
}

miniconda_conda <- function(path = miniconda_path()) {
  exe <- if (is_windows()) "condabin/conda.bat" else "bin/conda"
  conda <- file.path(path, exe)
  conda_binary(conda)
}

miniconda_envpath <- function(env, path = miniconda_path()) {
  file.path(path, "envs", env)
}

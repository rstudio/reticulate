
#' Install Miniconda
#'
#' Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
#' installer, and use it to install Miniconda.
#'
#' @param path The path in which Miniconda will be installed. Note that the
#'   installer does not support paths containing spaces.
#' 
#' @param update Boolean; update to the latest version of Miniconda after install?
#' 
#' @param force Boolean; force re-installation if Miniconda is already installed
#'   at the requested path?
#' 
#' @family miniconda
#' 
#' @export
install_miniconda <- function(path = miniconda_path(),
                              update = TRUE,
                              force = FALSE)
{
  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install Miniconda into a path containing spaces")
  
  # TODO: what behavior when miniconda is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniconda_preflight(path, force)
  
  # download the installer
  url <- miniconda_installer_url()
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
  python <- miniconda_python_package()
  conda_create("r-reticulate", packages = c(python, "numpy"), conda = conda)
  
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
  file.path(path, exe)
}

miniconda_envpath <- function(env, path = miniconda_path()) {
  file.path(path, "envs", env)
}

miniconda_meta_path <- function() {
  root <- rappdirs::user_data_dir("r-reticulate")
  file.path(root, "miniconda.json")
}

miniconda_meta_read <- function() {
  
  path <- miniconda_meta_path()
  if (!file.exists(path))
    return(list())
  
  json <- tryCatch(
    jsonlite::read_json(path),
    error = warning
  )
  
  if (is.list(json))
    return(json)
  
  list()
  
}

miniconda_meta_write <- function(data) {
  path <- miniconda_meta_path()
  dir.create(dirname(path), recursive = TRUE)
  json <- jsonlite::toJSON(data, auto_unbox = TRUE, pretty = TRUE)
  writeLines(json, con = path)
}

miniconda_installable <- function() {
  meta <- miniconda_meta_read()
  !identical(meta$DisableInstallationPrompt, TRUE)
}

miniconda_install_prompt <- function() {
  
  if (!interactive())
    return(FALSE)
  
  text <- paste(
    "No non-system installation of Python could be found.",
    "Would you like to download and install Miniconda?",
    "Miniconda is an open source environment management system for Python.",
    "See https://docs.conda.io/en/latest/miniconda.html for more details.",
    "",
    sep = "\n"
  )
  
  message(text)
  
  response <- readline("Would you like to install Miniconda? [Y/n]: ")
  repeat {
    
    ch <- tolower(substring(response, 1, 1))
    
    if (ch == "y" || ch == "") {
      install_miniconda()
      return(TRUE)
    }
    
    if (ch == "n") {
      
      meta <- miniconda_meta_read()
      meta$DisableInstallationPrompt <- TRUE
      miniconda_meta_write(meta)
      
      message("Installation aborted.")
      return(FALSE)
      
    }
    
    response <- readline("Please answer yes or no: ")
    
  }
  
}

# the default environment path to use for miniconda
miniconda_python_envpath <- function() {
  
  Sys.getenv(
    "RETICULATE_MINICONDA_PYTHON_ENVPATH",
    unset = miniconda_envpath("r-reticulate")
  )
  
}

# the version of python to use in the environment
miniconda_python_version <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PYTHON_VERSION", unset = "3.6")
}

miniconda_python_package <- function() {
  paste("python", miniconda_python_version(), sep = "=")
}

miniconda_enabled <- function() {
  
  enabled <- Sys.getenv("RETICULATE_MINICONDA_ENABLED", unset = "TRUE")
  if (tolower(enabled) %in% c("false", "0"))
    return(FALSE)
  
  miniconda_installable()
  
}

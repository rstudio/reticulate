
#' Install Miniconda
#'
#' Download the [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
#' installer, and use it to install Miniconda.
#'
#' @param path The path in which Miniconda will be installed. Note that the
#'   installer does not support paths containing spaces. See [miniconda_path]
#'   for more details on the default path used by `reticulate`.
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
  check_forbidden_install("Miniconda")
  
  if (grepl(" ", path, fixed = TRUE))
    stop("cannot install Miniconda into a path containing spaces")
  
  # TODO: what behavior when miniconda is already installed?
  # fail? validate installed and matches request? reinstall?
  install_miniconda_preflight(path, force)
  
  # download the installer
  url <- miniconda_installer_url()
  installer <- miniconda_installer_download(url)
  
  # run the installer
  message("* Installing Miniconda -- please wait a moment ...")
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
  
  url <- getOption("reticulate.miniconda.url")
  if (!is.null(url))
    return(url)
  
  # TODO: miniconda does not yet have arm64 binaries for macOS,
  # so we'll just use miniforge instead
  info <- as.list(Sys.info())
  if (info$sysname == "Darwin" && info$machine == "arm64") {
    base <- "https://github.com/conda-forge/miniforge/releases/latest/download"
    name <- "Miniforge3-MacOSX-arm64.sh"
    return(file.path(base, name))
  }

  base <- "https://repo.anaconda.com/miniconda"
  
  info <- as.list(Sys.info())
  arch <- miniconda_installer_arch(info)
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
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
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
  
  status <- system2(installer, args)
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

miniconda_envpath <- function(env = NULL, path = miniconda_path()) {
  env <- env %||% Sys.getenv("RETICULATE_MINICONDA_ENVNAME", unset = "r-reticulate")
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
  
  if (!is_interactive())
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
    unset = miniconda_envpath()
  )
  
}

# the version of python to use in the environment
miniconda_python_version <- function() {
  Sys.getenv("RETICULATE_MINICONDA_PYTHON_VERSION", unset = "3.8")
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

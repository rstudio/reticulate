
pyenv_root <- function() {
  root <- rappdirs::user_data_dir("r-reticulate")
  dir.create(root, showWarnings = FALSE, recursive = TRUE)
  norm <- normalizePath(root, winslash = "/", mustWork = TRUE)
  file.path(norm, "pyenv")
}

pyenv_python <- function(version) {

  if (is.null(version))
    return(NULL)
  
  # on Windows, Python will be installed as part of the pyenv installation
  prefix <- if (is_windows()) {
    pyenv <- pyenv_find()
    file.path(pyenv, "../../versions", version)
  } else {
    root <- Sys.getenv("PYENV_ROOT", unset = "~/.pyenv")
    file.path(root, "versions", version)
  }
  
  if (!file.exists(prefix)) {
    
    fmt <- paste(
      "Python %s does not appear to be installed.",
      "Try installing it with install_python(version = %s).",
      sep = "\n"
    )
    
    msg <- sprintf(fmt, version, shQuote(version))
    stop(msg)
    
  }
  
  stem <- if (is_windows()) "python.exe" else "bin/python"
  
  normalizePath(
    file.path(prefix, stem),
    winslash = "/",
    mustWork = TRUE
  )
  
}

pyenv_list <- function(pyenv = NULL) {
  
  # resolve pyenv
  pyenv <- normalizePath(
    pyenv %||% pyenv_find(),
    winslash = "/",
    mustWork = TRUE
  )
  
  # request list of Python packages
  output <- system2(pyenv, c("install", "--list"), stdout = TRUE, stderr = TRUE)
  
  # clean up output
  versions <- tail(output, n = -1L)
  cleaned <- gsub("^\\s*", "", versions)
  
  # only include CPython interpreters for now
  grep("^[[:digit:]]", cleaned, value = TRUE)
  
}

pyenv_find <- function() {
  pyenv <- pyenv_find_impl()
  normalizePath(pyenv, winslash = "/", mustWork = TRUE)
}

pyenv_find_impl <- function() {
  
  # check for pyenv binary specified via option
  pyenv <- getOption("reticulate.pyenv", default = NULL)
  if (!is.null(pyenv) && file.exists(pyenv))
    return(pyenv)
  
  # check for pyenv executable on the PATH
  pyenv <- Sys.which("pyenv")
  if (nzchar(pyenv))
    return(pyenv)
  
  # form stem path to pyenv binary (it differs between pyenv and pyenv-win)
  stem <- if (is_windows()) "pyenv-win/bin/pyenv" else "bin/pyenv"
  
  # check for a binary in the PYENV_ROOT folder
  root <- Sys.getenv("PYENV_ROOT", unset = "~/.pyenv")
  pyenv <- file.path(root, stem)
  if (file.exists(pyenv))
    return(pyenv)
  
  # check for reticulate's own pyenv
  root <- pyenv_root()
  pyenv <- file.path(root, stem)
  if (file.exists(pyenv))
    return(pyenv)
  
  # all else fails, try to manually install pyenv
  pyenv_bootstrap()
  
}

pyenv_install <- function(version, force, pyenv = NULL) {
  
  pyenv <- normalizePath(
    pyenv %||% pyenv_find(),
    winslash = "/",
    mustWork = TRUE
  )
  
  # set options
  withr::local_envvar(PYTHON_CONFIGURE_OPTS = "--enable-shared")
  
  # build install arguments
  force <- if (force)
    "--force"
  else if (!is_windows())
    "--skip-existing"
  
  args <- c("install", force, version)
  system2(pyenv, args)
  
}

pyenv_bootstrap <- function() {
  if (is_windows())
    pyenv_bootstrap_windows()
  else
    pyenv_bootstrap_unix()
}

pyenv_bootstrap_windows <- function() {
  
  # get path to pyenv
  root <- normalizePath(
    pyenv_root(),
    winslash = "/",
    mustWork = FALSE
  )
  
  # clone if necessary
  if (!file.exists(root)) {
    url <- "https://github.com/pyenv-win/pyenv-win"
    system2("git", c("clone", shQuote(url), shQuote(root)))
  }
  
  # ensure up-to-date
  owd <- setwd(root)
  on.exit(setwd(owd), add = TRUE)
  system("git pull")
  
  # return path to pyenv binary
  file.path(root, "pyenv-win/bin/pyenv")
  
}

pyenv_bootstrap_unix <- function() {
  
  if (!nzchar(Sys.which("git")))
    stop("bootstrapping of pyenv requires git to be installed")
  
  # move to tempdir
  owd <- setwd(tempdir())
  on.exit(setwd(owd), add = TRUE)
  
  # download the installer
  url <- "https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer"
  name <- basename(url)
  download.file(url, destfile = name, mode = "wb")
  
  # ensure it's runnable
  Sys.chmod(name, mode = "0755")
  
  # set root directory
  root <- pyenv_root()
  withr::local_envvar(PYENV_ROOT = root)
  
  # run the script
  message("Installing pyenv ...")
  status <- system("./pyenv-installer", intern = TRUE)
  if (!identical(status, 0L))
    stop("installation of pyenv failed")
  message("Done! pyenv successfully installed.")
  
  # return pyenv path
  file.path(root, "bin/pyenv")
  
}

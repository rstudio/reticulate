
pyenv_root <- function() {
  root <- rappdirs::user_data_dir("r-reticulate")
  file.path(root, "pyenv")
}

pyenv_list <- function(pyenv = NULL) {
  
  # resolve pyenv
  pyenv <- pyenv %||% pyenv_find()
  
  # request list of Python packages
  output <- system2(pyenv, c("install", "--list"), stdout = TRUE, stderr = TRUE)
  
  # clean up output
  versions <- tail(output, n = -1L)
  gsub("^\\s*", "", versions)
  
}

pyenv_find <- function() {
  
  # check for pyenv binary specified via option
  pyenv <- getOption("reticulate.pyenv", default = NULL)
  if (!is.null(pyenv) && file.exists(pyenv))
    return(pyenv)
  
  # check for pyenv executable on the PATH
  pyenv <- Sys.which("pyenv")
  if (nzchar(pyenv))
    return(pyenv)
  
  # check for a binary in the PYENV_ROOT folder
  root <- Sys.getenv("PYENV_ROOT", unset = "~/.pyenv")
  pyenv <- file.path(root, "bin/pyenv")
  if (file.exists(pyenv))
    return(pyenv)
  
  # check for reticulate's own pyenv
  root <- pyenv_root()
  pyenv <- file.path(root, "bin/pyenv")
  if (file.exists(pyenv))
    return(pyenv)
  
  # all else fails, try to manually install pyenv
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
  withr::local_envvar(PYENV_ROOT = pyenv_root())
  
  # run the script
  writeLines("Installing pyenv ...")
  system("./pyenv-installer")
  writeLines("Done!")
  
  # return pyenv path
  pyenv
  
}

pyenv_install <- function(version, force, pyenv = NULL) {
  
  pyenv <- pyenv %||% pyenv_find()
  
  # set options
  withr::local_envvar(PYTHON_CONFIGURE_OPTS = "--enable-shared")
  
  # build install arguments
  args <- c(
    "install",
    if (force) "--force" else "--skip-existing",
    version
  )
  
  system2(pyenv, args)
  
}

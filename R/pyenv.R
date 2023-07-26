
pyenv_root <- function() {
  root <- rappdirs::user_data_dir("r-reticulate")
  dir.create(root, showWarnings = FALSE, recursive = TRUE)
  norm <- normalizePath(root, winslash = "/", mustWork = TRUE)
  file.path(norm, "pyenv")
}


pyenv_resolve_latest_patch <- function(version, installed = TRUE, pyenv = pyenv_find()) {
  # if version = "3.8:latest", resolve latest installed
  # recent versions of pyenv on Mac/Linux can accept a ":latest" suffix on install,
  # but then we can't easily resolve the latest from what's locally installed.
  # windows pyenv can't handle :latest at all.
  # So we do it all in R.
  stopifnot(endsWith(version, ":latest"))
  version <- substr(version, 1L, nchar(version) - nchar(":latest"))

  available <- pyenv_list(pyenv, installed)
  available <- available[startsWith(available, version)]
  out <- as.character(max(numeric_version(available, strict = FALSE),
                          na.rm = TRUE))

  if (!length(out))
    stop(
      sprintf("Python release version '%s' not found. ", version),
      "Run `install_python(list = TRUE)` to see available versions."
    )

  out
}

pyenv_python <- function(version) {

  if (is.null(version))
    return(NULL)

  if (endsWith(version, ":latest"))
    version <- pyenv_resolve_latest_patch(version, installed = TRUE)

  # on Windows, Python will be installed as part of the pyenv installation
  prefix <- if (is_windows()) {
    pyenv <- pyenv_find()
    if (endsWith(pyenv, ".cmd")) {
      # check if it's a scoop shim, resolve the actual installation if so
      if (file.exists(actual_pyenv <-
                      str_drop_prefix(readLines(pyenv, 1), "@rem ")))
        pyenv <- actual_pyenv
    }
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

pyenv_list <- function(pyenv = NULL, installed = FALSE) {

  # resolve pyenv
  pyenv <- normalizePath(
    pyenv %||% pyenv_find(),
    winslash = "/",
    mustWork = TRUE
  )

  if (installed)
    return(system2(pyenv, c("versions", "--bare"), stdout = TRUE))

  # request list of Python packages
  output <- system2(pyenv, c("install", "--list"), stdout = TRUE, stderr = TRUE)

  # clean up output
  # on some platforms, warnings from cmd.exe appear in the output
  # also, there is a header like ":: [Info] ::  Mirror: https://www.python.org/ftp/python"
  # https://github.com/rstudio/reticulate/issues/1390
  header_end <- max(1L, grep("^:: \\[Info\\] :: .+$", output))
  versions <- tail(output, n = -header_end)
  cleaned <- gsub("^\\s*", "", versions)

  # only include CPython interpreters for now
  grep("^[[:digit:]]", cleaned, value = TRUE)

}

pyenv_find <- function(install = TRUE) {
  pyenv <- pyenv_find_impl(install = install)
  if (isFALSE(install) && is.null(pyenv))
    return(NULL)
  canonical_path(pyenv)
}

pyenv_find_impl <- function(install = TRUE) {

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
  if(install)
    pyenv_bootstrap()
  else
    NULL

}

pyenv_install <- function(version, force, pyenv = NULL) {

  pyenv <- canonical_path(pyenv %||% pyenv_find())
  stopifnot(file.exists(pyenv))

  # set options
  withr::local_envvar(PYTHON_CONFIGURE_OPTS = "--enable-shared")

  # build install arguments
  force <- if (force)
    "--force"
  else if (!is_windows())
    "--skip-existing"

  args <- c("install", force, version)
  system2t(pyenv, args)

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

  if (Sys.which("git") == "")
    stop("Please install git and ensure it is on your PATH")

  # clone if necessary
  if (!file.exists(root)) {
    url <- "https://github.com/pyenv-win/pyenv-win"
    system2("git", c("clone", shQuote(url), shQuote(root)))
  }

  # ensure up-to-date
  owd <- setwd(root)
  on.exit(setwd(owd), add = TRUE)
  system("git pull")

  # path to pyenv binary
  pyenv <- file.path(root, "pyenv-win/bin/pyenv")

  # running 'update' after install on windows is basically required
  # https://github.com/pyenv-win/pyenv-win/issues/280#issuecomment-1045027625
  system2(pyenv, "update")

  # return path to pyenv binary
  pyenv
}

pyenv_bootstrap_unix <- function() {

  if (!nzchar(Sys.which("git")))
    stop("bootstrapping of pyenv requires git to be installed")

  # move to tempdir
  owd <- setwd(tempdir())
  on.exit(setwd(owd), add = TRUE)

  # pyenv python builds are substantially faster on macOS if we pre-install
  # some dependencies (especially openssl) as pre-built but "untapped kegs"
  # (i.e., unlinked to somewhere on the PATH but tucked away under $BREW_ROOT/Cellar).
  if (nzchar(Sys.which("brew"))) {
    system2t("brew", c("install -q openssl readline sqlite3 xz zlib tcl-tk"))
    system2t("brew", c("install --only-dependencies pyenv python"))
  }

  # download the installer
  url <- "https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer"
  name <- basename(url)
  download.file(url, destfile = name, mode = "wb")

  # ensure it's runnable
  Sys.chmod(name, mode = "0755")

  # set root directory
  root <- pyenv_root()
  withr::local_envvar(PYENV_ROOT = root)

  # run the script -- for some reason, the pyenv installer will return
  # a non-zero status code even on success?
  writeLines("Installing pyenv ...")

  suppressWarnings(system("./pyenv-installer", intern = TRUE))
  path <- file.path(root, "bin/pyenv")
  if (!file.exists(path))
    stop("installation of pyenv failed")

  writeLines(sprintf("Done! pyenv has been installed to '%s'.", path))
  path

}


pyenv_update <- function(pyenv = pyenv_find()) {

  if (startsWith(pyenv, root <- pyenv_root())) {
    # this pyenv installation is fully managed by reticulate
    # root == where .../bin/pyenv lives
    withr::with_dir(root, system2("git", "pull", stdout = FALSE, stderr = FALSE))
  }

  if (is_windows())
    return(system2t(pyenv, "update"))

  # $ git clone https://github.com/pyenv/pyenv-update.git $(pyenv root)/plugins/pyenv-update
  # root == ~/.pyenv == where installed pythons live
  root <- system2(pyenv, "root", stdout = TRUE)
  if (!dir.exists(file.path(root, "plugins/pyenv-update")))
    system2("git", c("clone", "https://github.com/pyenv/pyenv-update.git",
                      file.path(root, "plugins/pyenv-update")))

  result <- system2t(pyenv, "update", stdout = FALSE, stderr = FALSE)
  if (!identical(result, 0L)) {
    fmt <- "Error updating pyenv [exit code %i]"
    warningff(fmt, result)
  }

}

#export PATH="$HOME/.local/share/r-reticulate/pyenv/bin/:$PATH"
#eval "$(pyenv init --path)"
#eval "$(pyenv virtualenv-init -)"

#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init --path)"
#eval "$(pyenv virtualenv-init -)"

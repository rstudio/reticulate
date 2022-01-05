
python_has_modules <- function(python, modules) {

  # write code to tempfile
  file <- tempfile("reticulate-python-", fileext = ".py")
  code <- paste("import", modules)
  writeLines(code, con = file)
  on.exit(unlink(file), add = TRUE)

  # invoke Python
  status <- system2(python, shQuote(file), stdout = FALSE, stderr = FALSE)
  status == 0L

}

python_has_module <- function(python, module) {
  code <- paste("import", module)
  args <- c("-E", "-c", shQuote(code))
  status <- system2(python, args, stdout = FALSE, stderr = FALSE)
  status == 0L
}

python_version <- function(python) {
  code <- "import platform; print(platform.python_version())"
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = FALSE)
  sanitized <- gsub("[^0-9.-]", "", output)
  numeric_version(sanitized)
}

python_module_version <- function(python, module) {
  fmt <- "import %1$s; print(%1$s.__version__)"
  code <- sprintf(fmt, module)
  args <- c("-E", "-c", shQuote(code))
  output <- system2(python, args, stdout = TRUE, stderr = FALSE)
  numeric_version(output)
}

# given the path to a Python binary associated with a Python virtual environment
# or a Python Conda environment, try to ascertain its type
python_info <- function(python) {

  path <- dirname(python)
  parent <- dirname(path)

  # NOTE: we check for both 'python' and 'python3' because certain python
  # installations might install one version of the binary but not the other.
  #
  # Some installations might not place Python within a 'Scripts' or 'bin'
  # sub-directory, so look in the root directory too.
  prefixes <- list(NULL, if (is_windows()) "Scripts" else "bin")
  suffixes <- if (is_windows()) "python.exe" else c("python", "python3")

  while (path != parent) {

    # check for virtual environment files
    files <- c(
      "pyvenv.cfg",                                  # created by venv
      file.path(prefixes[[2L]], "activate_this.py")  # created by virtualenv
    )

    paths <- file.path(path, files)
    virtualenv <- any(file.exists(paths))

    # extra check that we aren't in a conda environment for Windows
    if (is_windows()) {
      condapath <- file.path(path, "condabin/conda")
      if (file.exists(condapath))
        virtualenv <- FALSE
    }

    if (virtualenv)
      return(python_info_virtualenv(path))

    # check for conda environment files
    condaenv <-
      file.exists(file.path(path, "conda-meta")) &&
      !file.exists(file.path(path, "condabin"))

    if (condaenv)
      return(python_info_condaenv(path))

    # check for python binary (implies a system install)
    for (prefix in prefixes) {
      for (suffix in suffixes) {
        bin <- paste(c(path, prefix, suffix), collapse = "/")
        if (file.exists(bin))
          return(python_info_system(path, bin))
      }
    }

    # recurse
    parent <- path
    path <- dirname(path)

  }

  stopf("could not find a Python environment for %s", python)

}

python_info_virtualenv <- function(path) {

  # form path to python binary
  suffix <- if (is_windows()) "Scripts/python.exe" else "bin/python"
  python <- file.path(path, suffix)

  # return details
  list(
    python = python,
    type = "virtualenv",
    root = path
  )

}

python_info_condaenv <- function(path) {

  # form path to python binary
  suffix <- if (is_windows()) "python.exe" else "bin/python"
  python <- file.path(path, suffix)

  # find path to conda associated with this env
  conda <- python_info_condaenv_find(path)

  list(
    python = python,
    type   = "conda",
    root   = path,
    conda  = conda
  )

}

python_info_condaenv_find <- function(path) {

  # read history file
  histpath <- file.path(path, "conda-meta/history")
  if (!file.exists(histpath))
    return(NULL)

  history <- readLines(histpath, warn = FALSE)

  # look for cmd line
  pattern <- "^[[:space:]]*#[[:space:]]*cmd:[[:space:]]*"
  lines <- grep(pattern, history, value = TRUE)
  if (length(lines) == 0)
    return(NULL)

  # get path to conda script used
  line <- gsub(pattern, "", lines[[1]])
  parts <- strsplit(line, "[[:space:]]+")[[1]]
  script <- parts[[1]]

  # on Windows, a wrapper script is recorded in the history,
  # so instead attempt to find the real conda binary
  exe <- if (is_windows()) "conda.exe" else "conda"
  conda <- file.path(dirname(script), exe)

  normalizePath(conda, winslash = "/", mustWork = FALSE)

}

python_info_system <- function(path, python) {

  list(
    python = python,
    type   = "system",
    root   = path
  )

}

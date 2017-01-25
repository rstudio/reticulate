
# TODO: test w/ various versions of python
# TODO: scanning for versions not explicitly known

# https://github.com/JuliaPy/PyCall.jl/blob/master/deps/build.jl
# https://docs.python.org/3/c-api/init.html#c.Py_SetProgramName


python_config <- function() {

  # determine the location of python
  tensorflow_python <- Sys.getenv("TENSORFLOW_PYTHON", unset = NA)
  if (!is.na(tensorflow_python)) {

    # normalize trailing slash
    tensorflow_python <- gsub("[\\/]+$", "", tensorflow_python)

    # check for existence
    if (!file_test("-d", tensorflow_python) && !file_test("-f", tensorflow_python))
      stop("Specified TENSORFLOW_PYTHON (", tensorflow_python, ") does not exist.")

    # set python var
    python <- tensorflow_python

    # append binary if it's a directory
    if (file_test("-d", python))
      python <- file.path(python, "python")

    # append .exe if necessary on windows
    if (is_windows() && (!endsWith(tolower(python), ".exe")))
      python <- paste0(python, ".exe")

  } else {

    # find system python and verify that it exists
    python <- Sys.which("python")
    if (!nzchar(python))
      stop("No version of python found on system PATH. Please ensure python is on your ",
           "path or specify the location using the TENSORFLOW_PYTHON environment variable")
  }

  # helper to execute python code and return stdout
  exec_python <- function(command) {
    system(command = paste(shQuote(python), "-c", shQuote(command)), intern = TRUE)
  }

  py_config_var <- function(var) {
    exec_python(sprintf("import sys; import sysconfig; sys.stdout.write(sysconfig.get_config_vars('%s')[0]);",
                        var))
  }

  # determine the version
  version <- exec_python("import sys; sys.stdout.write(str(sys.version_info.major) + '.' + str(sys.version_info.minor));")

  # determine the location of libpython
  if (is_windows()) {
    # note that 'prefix' has the binary location and 'py_version_nodot` has the suffix`
    python_libdir <- dirname(python)
    libpython <- file.path(python_libdir, paste0("python", gsub(".", "", version, fixed = TRUE), ".dll"))
  } else {
    # (note that the LIBRARY variable has the name of the static library)
    python_libdir <- py_config_var('LIBPL')
    ext <- switch(Sys.info()[["sysname"]], Darwin = ".dylib", Windows = ".dll", ".so")
    libpython <- file.path(python_libdir, paste0("libpython", version, ext))
  }
  if (!file.exists(libpython))
    stop("Python shared library '", libpython, "' not found.")

  # determine PYTHONHOME
  pythonhome <- normalizePath(py_config_var("prefix"), mustWork = FALSE)
  if (!is_windows())
    pythonhome <- paste(pythonhome,
                        normalizePath(py_config_var("exec_prefix"), mustWork = FALSE),
                        sep = ":")

  # return config info
  structure(class = "python_config", list(
    python = normalizePath(python),
    libpython = normalizePath(libpython),
    pythonhome = pythonhome,
    version = numeric_version(version)
  ))

}

#' @export
print.python_config <- function(x, ...) {
  cat("python:     ", x$python, "\n")
  cat("libpython:  ", x$libpython, "\n")
  cat("pythonhome: ", x$pythonhome, "\n")
  cat("version:    ", format(x$version), "\n")
}


is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}



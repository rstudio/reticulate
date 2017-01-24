
# TODO: test w/ various versions of python
# TODO: scanning for versions not explicitly known

python_config <- function() {

  # determine the location of python
  tensorflow_python <- Sys.getenv("TENSORFLOW_PYTHON", unset = NA)
  if (!is.na(tensorflow_python)) {
    if (!file.exists(tensorflow_python))
      stop("Specified TENSORFLOW_PYTHON (", tensorflow_python, ") does not exist.")
    python <- tensorflow_python
    if (file_test("-d", python))
      python <- file.path(python, "python")
  } else {
    python <- Sys.which("python")
    if (!nzchar(python))
      stop("No version of python found on system PATH. Please ensure python is on your ",
           "path or specify the location using the TENSORFLOW_PYTHON environment variable")
  }

  # helper to execute python code and return stdout
  exec_python <- function(command) {
    system(command = paste(shQuote(python), "-c", shQuote(command)), intern = TRUE)
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
    python_libdir <- exec_python("import sys; import sysconfig; sys.stdout.write(sysconfig.get_config_vars('LIBPL')[0]);")
    ext <- switch(Sys.info()[["sysname"]], Darwin = ".dylib", Windows = ".dll", ".so")
    libpython <- file.path(python_libdir, paste0("libpython", version, ext))
  }

  # return config info
  structure(class = "python_config", list(
    python = python,
    libpython = libpython,
    version = numeric_version(version)
  ))

}

#' @export
print.python_config <- function(x, ...) {
  cat("python:     ", x$python, "\n")
  cat("version:    ", format(x$version), "\n")
  cat("libpython:  ", x$libpython, "\n")
}


is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}



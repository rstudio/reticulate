
# Python configuration
#
# Retreive a list containing Python configuration information including the
# \code{python} binary and \code{libpython} the package was built against and
# the location of the \code{numpy} include directory the package was built
# against.
py_config <- function() {

  # extract full path to libpython from PKG_LIBS
  match <- regexpr("-L[^ ]+", PKG_LIBS)
  if (match != -1) {
    libpython_dir <- regmatches(PKG_LIBS, match)
    libpython_dir <- substring(libpython_dir, 3)
  } else {
    libpython_dir <- NULL
  }

  # extract python library from PKG_LIBS
  match <- regexpr("-lpython\\d+\\.?\\d+", PKG_LIBS)
  if (match == -1)
    stop("Unable to parse -lpython from ", PKG_LIBS)
  libpython_lib <- regmatches(PKG_LIBS, match)
  libpython_lib <- substring(libpython_lib, 3)
  ext <- switch(Sys.info()[["sysname"]],
    Darwin = ".dylib",
    Windows = ".dll",
    ".so"
  )
  libpython_lib <- paste0(ifelse(ext != ".dll","lib", ""),
                          libpython_lib,
                          ext)

  # provide full path to libpython if we have a dir
  if (!is.null(libpython_dir)) {
    libpython <- file.path(libpython_dir, libpython_lib)
  } else {
    libpython <- libpython_lib
  }

  # return configuration
  list(python = PYTHON_BIN,
       libpython = libpython,
       numpy = NUMPY_INCLUDE_DIR)
}


#' TensorFlow configuration information
#'
#' Retreive a list containing configuration information including the
#' \code{python} binary and \code{libpython} the package was built against,
#' , and the version and location of the TensorFlow python module.
#'
#' @export
tf_config <- function() {

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

  # build configuration
  config <- structure(class = "tf_config", list(
    python = PYTHON_BIN,
    libpython = libpython)
  )

  # add TF config info
  if (!is.null(tf)) {
    config$tf <- normalizePath(tf$`__path__`, winslash = "/")
    config$tf_version <- tf$`__version__`
  }

  # return config
  config
}

#' @export
print.tf_config <- function(x, ...) {
  cat("python:     ", x$python, "\n")
  cat("libpython:  ", x$libpython, "\n")
  if (!is.null(x$tf)) {
    cat("tf:         ", x$tf, "\n")
    cat("tf_version: ", x$tf_version, "\n")
  }
}



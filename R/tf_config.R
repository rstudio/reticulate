
#' TensorFlow configuration information
#'
#' Retreive a list containing configuration information including the
#' \code{python} binary and \code{libpython} the package was built against,
#' , and the version and location of the TensorFlow python module.
#'
#' @export
tf_config <- function() {

  # build configuration
  config <- python_config()
  if (!is.null(tf)) {
    config$tf <- normalizePath(tf$`__path__`, winslash = "/")
    config$tf_version <- tf$`__version__`
  }
  class(config) <- c("tf_config", "python_config")

  # return config
  config
}

#' @export
print.tf_config <- function(x, ...) {
  print.python_config(x, ...)
  if (!is.null(x$tf)) {
    cat("tf:         ", x$tf, "\n")
    cat("tf_version: ", x$tf_version, "\n")
  }
}



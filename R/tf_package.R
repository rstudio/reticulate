
# record of load error message
.load_error_message <- NULL

tf_on_load <- function(libname, pkgname) {

  # attempt to load tensorflow
  tf <<- tryCatch(import("tensorflow"), error = function(e) e)
  if (inherits(tf, "error")) {
    .load_error_message <<- tf$message
    tf <<- NULL
  }

  # if we loaded tensorflow then register tf help topics
  if (!is.null(tf))
    register_tf_help_topics()
}

tf_on_attach <- function(libname, pkgname) {
  if (is.null(tf)) {
    packageStartupMessage(.load_error_message)
    packageStartupMessage("If you have not yet installed TensorFlow, see ",
                          "https://www.tensorflow.org/get_started/")
    packageStartupMessage("Configuration:")
    config <- py_config()
    packageStartupMessage(" python:    ", config$python)
    packageStartupMessage(" libpython: ", config$libpython)
    packageStartupMessage(" numpy:     ", config$numpy)
  }
}

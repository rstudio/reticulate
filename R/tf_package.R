
tf_on_load <- function(libname, pkgname) {
  # attempt to load tensorflow
  tf <<- import("tensorflow", silent = TRUE)

  # if we loaded tensorflow then register tf help topics
  if (!is.null(tf))
    register_tf_help_topics()
}

tf_on_attach <- function(libname, pkgname) {
  if (is.null(tf)) {
    packageStartupMessage("TensorFlow not currently installed, please see ",
                          "https://www.tensorflow.org/get_started/")
  }
}

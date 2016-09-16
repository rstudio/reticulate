
# TODO: function for reading MNIST data

# TODO: Non-integer warning when reading MNIST data (try w/o flags):
#   /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:164:
#   VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error
#   in the future
#    return self._images[start:end], self._labels[start:end]

# TODO: there is some state for the summary early in the training
#       which we can get into which causes a proto write error
#       (just got it again at 13700 iterations, yikes!). Perhaps
#       a NaN that gets marshalled from R in an unexpected fashion?

#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {

  # initialize python
  config <- py_config()
  py_initialize(config$libpython);

  # add our python scripts to the search path
  py_run_string(paste0("import sys; sys.path.append('",
                       system.file("python", package = "tensorflow") ,
                       "')"))

  # call tf onLoad handler
  tf_on_load(libname, pkgname)
}


.onAttach <- function(libname, pkgname) {
  # call tf onAttach handler
  tf_on_attach(libname, pkgname)
}

.onUnload <- function(libpath) {
  py_finalize();
}


# TODO: forward full path to python lib for dlopen
# TODO: add python superclasses
# TODO: implements with or with_context:
#       http://preshing.com/20110920/the-python-with-statement-by-example/
# TODO: generally error handling

#' @useDynLib tensorflow
#' @importFrom Rcpp evalCpp
NULL

# package level tf instance
tf <- NULL

.onLoad <- function(libname, pkgname) {

  # initialize python
  py_initialize(pythonSharedLibrary());

  # attempt to load tensorflow
  tf <<- tryCatch(tensorflow(), error = function(e) NULL)
}


.onAttach <- function(libname, pkgname) {
  if (!is_tensorflow_installed()) {
    packageStartupMessage("TensorFlow not currently installed, please see ",
                          "https://www.tensorflow.org/get_started/")
  }
}

.onUnload <- function(libpath) {
  py_finalize();
}


#' Import TensorFlow
#'
#' Import the tensorflow python module (or one of it's sub-modules) for
#' use in R.
#'
#' @param module Name of sub-module to import. Defaults to \code{NULL}, which
#' imports the main tensorflow module.
#'
#' @return A tensorflow module
#'
#' @examples
#' \dontrun{
#' tf <- tensorflow()
#' tflearn <- tensorflow("contrib.learn")
#' slim <- tenstensorflow("contrib.slim")
#' }
#'
#' @export
tensorflow <- function(module = NULL) {
  if (is.null(module))
    py_import("tensorflow")
  else
    py_import(paste("tensorflow", module, sep="."))
}

#' @rdname tensorflow
#' @export
is_tensorflow_installed <-function() {
  !is.null(tf)
}

#' Tensor shape
#'
#' @param ... Tensor dimensions
#'
#' @export
shape <- function(...) {
  dims <- list(...)
  lapply(dims, function(dim) {
    if (!is.null(dim))
      as.integer(dim)
    else
      NULL
  })
}

#' @export
"print.tensorflow.python.framework.ops.Tensor" <- function(x, ...) {
  print.py_object(x, ...)
  if (!is.null(tf$get_default_session())) {
    value <- tryCatch(x$eval(), error = function(e) NULL)
    if (!is.null(value))
      cat(" ", str(value), "\n", sep = "")
  }
}

#' @export
"+.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$add(a, b)
}

#' @export
"+.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$add(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"-.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  if (missing(b))
    tf$neg(a)
  else
    tf$sub(a, b)
}

#' @export
"*.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$mul(a, b)
}

#' @export
"*.tensorflow.python.ops.variables.Variable" <- function(a, b) {
  tf$mul(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$truediv(a, b)
}

#' @export
"/.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$truediv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$floordiv(a, b)
}

#' @export
"%/%.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$floordiv(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$mod(a, b)
}

#' @export
"%%.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$mod(a, b)
}

#' @export
"^.tensorflow.python.framework.ops.Tensor" <- function(a, b) {
  tf$pow(a, b)
}

#' @export
"^.tensorflow.python.framework.ops.Variable" <- function(a, b) {
  tf$pow(a, b)
}


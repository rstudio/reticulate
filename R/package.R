
# TODO: handle complex numbers
#
#
# TODO: there is some state for the summary early in the training
#       which we can get into which causes a proto write error
#       (just got it again at 13700 iterations, yikes!). Perhaps
#       a NaN that gets marshalled from R in an unexpected fashion?
#
#       Setting break on error in RStudio reveals that the summary string is:
#
#         - "\n\024\n\rxentropy_mean\025\xa7\xe6"
#
#       After doing the UTF-8 work we get this:
#         - "\n\024\n\rxentropy_mean\025"
#
#       Perhaps a unicode string we aren't handling properly
#
#  There is something going on with round-tripping of strings (if we
#  don't convert python strings to R then the problem goes away). A
#  possible solution:
#    - When converting a python string to R keep a map of string values
#      back to the python strings, then when converting from R to python
#      feed values from the map back to R when we can. This should work
#      because python strings are immutable.
#
#  Did some experimenting with string pools to no avail. We might also want
#  to create a mechanism by which certain methods get no conversion (e.g.
#  session$run)
#


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

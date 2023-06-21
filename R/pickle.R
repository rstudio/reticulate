

#' Save and Load Python Objects
#'
#' Save and load Python objects.
#'
#' Python objects are serialized using the `pickle` module -- see
#' <https://docs.python.org/3/library/pickle.html> for more details.
#'
#' @param object A Python object.
#'
#' @param filename The output file name. Note that the file extension `.pickle`
#'   is considered the "standard" extension for serialized Python objects
#'   as created by the `pickle` module.
#'
#' @param pickle The "pickle" implementation to use. Defaults to `"pickle`",
#'   but other compatible Python "pickle" implementations (e.g. `"cPickle"`)
#'   could be used as well.
#'
#' @param ... Optional arguments, to be passed to the `pickle` module's
#'   `dump()` and `load()` functions.
#'
#' @param convert Bool. Whether the loaded pickle object should be converted to
#'   an R object.
#'
#' @export
py_save_object <- function(object, filename, pickle = "pickle", ...) {

  filename <- normalizePath(filename, winslash = "/", mustWork = FALSE)

  builtins <- import_builtins()
  pickle <- import(pickle, convert = TRUE)

  handle <- builtins$open(filename, "wb")
  on.exit(handle$close(), add = TRUE)
  pickle$dump(object, handle, protocol = pickle$HIGHEST_PROTOCOL, ...)

}

#' @rdname py_save_object
#' @export
py_load_object <- function(filename, pickle = "pickle", ..., convert = TRUE) {

  filename <- normalizePath(filename, winslash = "/", mustWork = FALSE)

  builtins <- import_builtins()
  pickle <- import(pickle, convert = convert)

  handle <- builtins$open(filename, "rb")
  on.exit(handle$close(), add = TRUE)
  obj <- py_call(pickle$load, handle, ...)
  py_maybe_convert(obj, convert)
}


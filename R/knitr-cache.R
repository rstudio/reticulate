#' A reticulate cache engine for Knitr
#'
#' This provides a `reticulate` cache engine for `knitr`. The cache engine
#' allows `knitr` to save and load Python sessions between cached chunks. The
#' cache engine depends on the `dill` Python module. Therefore, you must have
#' `dill` installed in your Python environment.
#'
#' The engine can be activated by setting (for example)
#'
#' ```
#' knitr::cache_engines$set(python = reticulate::cache_eng_python)
#' ```
#'
#' Typically, this will be set within a document's setup chunk, or by the
#' environment requesting that Python chunks be processed by this engine.
#'
#' @param options
#'   List of chunk options provided by `knitr` during chunk execution.
#'   Contains the caching path.
#'
#' @export
cache_eng_python <- (function() {
  check_cache_available <- function() {
    # does the python version is supported by 'dill'?
    if (py_version() < "3.7") {
      warning("Python cache requires Python version >= 3.7")
      return(FALSE)
    }

    # is the module 'dill' loadable?
    dill <- tryCatch(import("dill"), error = identity)
    if (inherits(dill, "error")) {
      error <- reticulate::py_last_error()
      if (!error$type %in% c("ImportError", "ModuleNotFoundError"))
        stop(error$value, call. = FALSE)
      warning("The Python module 'dill' was not found, it's required for Python cache")
      return(FALSE)
    }

    # is the 'dill' version recent enough?
    dill_version <- as_numeric_version(dill$`__version__`)
    if (dill_version < "0.3.6") {
      warning("Python cache requires module dill>=0.3.6")
      return(FALSE)
    }

    # Python cache is available
    TRUE
  }

  cache_available <- function() {
    available <- knitr::opts_knit$get("reticulate.cache")
    if (is.null(available)) {
      available <- check_cache_available()
      knitr::opts_knit$set(reticulate.cache = available)
    }
    available
  }

  cache_path <- function(path) {
    paste(path, "pkl", sep=".")
  }

  cache_exists <- function(options) {
    file.exists(cache_path(options$hash))
  }

  cache_load <- function(options) {
    eng_python_initialize(options, envir = environment())
    if (!cache_available()) return()
    dill <- import("dill")
    dill$load_module(filename = cache_path(options$hash), module = "__main__")
  }

  filter <- NULL
  r_obj_filter <- function() {
    if (is.null(filter)) {
      filter <<- py_eval("lambda obj: obj.name == 'r' and type(obj.value) is __builtins__.__R__")
    }
    filter
  }

  cache_save <- function(options) {
    if (!cache_available()) return()
    dill <- import("dill")
    tryCatch({
      dill$dump_module(cache_path(options$hash), refimported = TRUE, exclude = r_obj_filter())
    }, error = function(e) {
      cache_purge(options$hash)
      stop(e)
    })
  }

  cache_purge <- function(glob_path) {
    unlink(cache_path(glob_path))
  }

  list(exists = cache_exists, load = cache_load, save = cache_save, purge = cache_purge)
})()

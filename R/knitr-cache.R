#' A reticulate cache engine for Knitr
#'
#' This provides caching of Python variables to the `reticulate` engine for
#' `knitr`. The cache allows `knitr` to save and load the state of Python
#' variables between cached chunks.  The cache engine depends on the `dill`
#' Python module. Therefore, you must have a recent version of `dill` installed
#' in your Python environment.
#'
#' The Python cache is activated the same way as the R cache, by setting the
#' `cache` chunk option to `TRUE`.  To _deactivate_ the Python cache globally
#' while keeping the R cache active, one may set the option `reticulate.cache`
#' to `FALSE`.  For example:
#'
#' ```
#' knitr::opts_knit$set(reticulate.cache = FALSE)
#' ```
#'
#' @note Different from `knitr`'s R cache, the Python cache is capable of saving
#' most, but not all types of Python objects.  Some Python objects are
#' "unpickleable" and will rise an error when attepmted to be saved.
#'
#' @export
cache_eng_python <- (function() {
  closure <- environment()
  dill <- NULL

  cache_path <- function(path) {
    paste(path, "pkl", sep=".")
  }

  check_cache_available <- function(options) {
    MINIMUM_PYTHON_VERSION <- "3.7"
    MINIMUM_DILL_VERSION <- "0.3.6"

    eng_python_initialize(options)

    # does the python version is supported by 'dill'?
    if (py_version() < MINIMUM_PYTHON_VERSION) {
      warning("Python cache requires Python version >= ", MINIMUM_PYTHON_VERSION)
      return(FALSE)
    }

    # is the module 'dill' loadable and recent enough?
    closure$dill <- tryCatch(import("dill"), error = identity)
    if (!inherits(dill, "error")) {
      dill_version <- as_numeric_version(dill$`__version__`)
      if (dill_version >= MINIMUM_DILL_VERSION)
        return(TRUE)
    } else {
      # handle non-import error
      error <- reticulate::py_last_error()
      if (!error$type %in% c("ImportError", "ModuleNotFoundError"))
        stop(error$value, call. = FALSE)
    }

    # 'dill' isn't available
    warning("Python cache requires module dill>=", MINIMUM_DILL_VERSION)
    FALSE
  }

  cache_available <- function(options) {
    if (is.null(closure$.cache_available))
      closure$.cache_available <- check_cache_available(options)
    .cache_available
  }

  cache_exists <- function(options) {
    file.exists(cache_path(options$hash))
  }

  cache_load <- function(options) {
    if (!cache_available(options)) return()
    dill$load_module(filename = cache_path(options$hash), module = "__main__")
  }

  cache_save <- function(options) {
    if (!cache_available(options)) return()

    # remove injected 'r' object before saving session (and after executing block)
    main <- import_main(convert = FALSE)
    if (py_has_attr(main, "r")) {
      builtins <- import_builtins(convert = TRUE)
      if (builtins$isinstance(main$r, builtins[["__R__"]]))
        py_del_attr(main, "r")
    }

    tryCatch({
      # refimported: save imported objects by reference when possible
      dill$dump_module(cache_path(options$hash), refimported = TRUE)
    }, error = function(e) {
      cache_purge(options$hash)
      stop(e)
    })
  }

  cache_purge <- function(glob_path) {
    unlink(cache_path(glob_path))
  }

  list(available = cache_available, exists = cache_exists, load = cache_load, save = cache_save,
       purge = cache_purge)
})()

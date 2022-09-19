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
        cache_initialize()
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

  cache_initialize <- function() {
    # save imported objects by reference when possible
    dill.session <- import("dill.session")
    dill.session[["settings"]][["refimported"]] <- TRUE
  }

  cache_exists <- function(options) {
    file.exists(cache_path(options$hash))
  }

  cache_load <- function(options) {
    if (!cache_available(options)) return()
    dill$load_module(filename = cache_path(options$hash), module = "__main__")
  }

  r_obj_filter <- function() {
    if (is.null(closure$.r_obj_filter)) {
      expr <- "lambda obj: obj.name == 'r' and type(obj.value) is __builtins__.__R__"
      closure$.r_obj_filter <- py_eval(expr)
    }
    .r_obj_filter
  }

  cache_save <- function(options) {
    if (!cache_available(options)) return()

    # when only inclusion filters are specified, it works as an allowlist
    if (!is.null(options$cache.vars)) {
      exclude <- NULL  # the R object won't be saved unless specified by cache.vars
      include <- options$cache.vars
    } else {
      exclude <- r_obj_filter()
      include <- NULL
    }

    tryCatch({
      dill$dump_module(cache_path(options$hash), exclude = exclude, include = include)
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

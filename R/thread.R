

#' Create a Python function that will always be called on the main thread
#'
#' This function is helpful when you need to provide a callback to a Python
#' library which may invoke the callback on a background thread. As R functions
#' must run on the main thread, wrapping the R function with `py_main_thread_func()`
#' will ensure that R code is only executed on the main thread.
#'
#' @param f An R function with arbitrary arguments
#' @return A Python function that delegates to the passed R function, which
#'  is guaranteed to always be called on the main thread.
#'
#'
#' @export
py_main_thread_func <- function(f) {
  r_to_py(f, TRUE) # every R func is a main thread func.
}


py_allow_threads <- function(allow = TRUE) {
  if (allow) {
    reticulate_ns <- environment(sys.function())
    for (f in sys.frames()) {
      if (identical(parent.env(f), reticulate_ns) &&
          !identical(f, environment()))
        # Can't release the gil as unlocked while we're holding it
        # elsewhere on the callstack.
        stop("Python threads can only be unblocked from a top-level reticulate call")
    }
  }

  if (!was_python_initialized_by_reticulate())
    stop("Can't safely unblock threads when R is running embedded")

  invisible(py_allow_threads_impl(allow))
}



## TODO: document how to use sys.unraisablehook() to customize handling of exceptions
## from background threads. Or, switch to using the threading module, which
## has more options for customizing exceptions hooks, and document that.
## TODO: give a meaningful name for the thread that appears in tracebacks.
## Either use the threading module and pass name=,
##   or do something like
##     f = lambda file: run_file(file)
##     f.__name__ = "run: " + os.path.basename(file)
py_run_file_on_thread <- function(file, ..., args = NULL) {
  if (!is.null(args))
    args <- as.list(as.character(args))
  import("rpytools.run")$`_launch_lsp_server_on_thread`(file, args)
}

## used in Positron:
# reticulate:::py_run_file_on_thread(
#   file = "${kernelPath}",
#   args = c(
#     "-f", "${connnectionFile}",
#     "--logfile", "${logFile}",
#     "--loglevel", "${logLevel}",
#     "--session-mode", "console"
#   )
# )

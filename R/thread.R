

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
  tools <- import("rpytools")
  tools$thread$main_thread_func(f)
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



py_run_file_on_thread <- function(file, ..., argv) {
  import("rpytools.run")$`_launch_lsp_server_on_thread`(file, as.list(as.character(argv)))
}

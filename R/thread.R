

#' Create a Python function that will always be called on the main thread
#'
#' This function is helpful when you need to provide a callback to a Python
#' library which may invoke the callback on a background thread. As R functions
#' must run on the main thread, wrapping the R function with `py_main_thread_func()`
#' will ensure that R code is only executed on the main thread.
#'
#' @param f An R function with artibrary arguments
#' @return A Python function that delegates to the passed R function, which
#'  is guaranteed to always be called on the main thread.
#'
#'
#' @export
py_main_thread_func <- function(f) {
  tools <- import("rpytools")
  tools$thread$main_thread_func(f)
}



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

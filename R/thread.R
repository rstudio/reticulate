

#' [Deprecated] Create a Python function that will always be called on the main thread
#'
#' Beginning with reticulate v1.39.0, every R function is a "main thread func". Usage of `py_main_thread_func()`
#' is no longer necessary.
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
#' @keywords internal
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

  # TODO: we should have a dedicated entry point in reticulate for this.
  # Needs to be updated in ark and positron.
  launching_lsp <- (basename(file) == 'positron_language_server.py' &&
                    is_positron() &&
                    # The basename changed to 'posit' in recent Positron 2025.03
                    basename(dirname(file)) %in% c("positron", "posit"))

  if (launching_lsp) {
    main_dict <- py_eval("__import__('__main__').__dict__.copy()", FALSE)
    py_get_attr(main_dict, "pop")("__annotations__")
    # IPykernel will create a thread that redirects all output from fileno of
    # the current sys.stdout and sys.stderr to its IO channels.
    # This is not correctly cleaned up when IPykernel closes.
    # To fix that, we set the IO streams to /dev/null before launching the kernel.
    import("rpytools.run")$set_blank_io_streams()
  }

  import("rpytools.run")$run_file_on_thread(file, args, ...)

  if (launching_lsp) {
    PositronIPKernelApp <- tryCatch(
      # NOTE: Update `import_positron_ipykernel_inspectors` when changing here
      import("positron.positron_ipkernel")$PositronIPKernelApp,
      error = function(err) {
        # Prior to Positron v2025.03 this was used to access PositronIPKernelApp
        import("positron_ipykernel.positron_ipkernel")$PositronIPKernelApp
      }
    )

    for(i in 1:40) { # Positron timeout is 20 seconds
      if (PositronIPKernelApp$initialized()) break
      Sys.sleep(.5)
    }
    Sys.sleep(1)

    py_eval("__import__('__main__').__dict__.update", FALSE)(main_dict)
  }
  invisible()
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

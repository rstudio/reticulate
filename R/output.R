remap_output_streams <- function() {
  # force remapping of output streams (required with Windows + Python 2
  # as otherwise output from Python is lost)
  force <- is_windows() && !is_python3()
  remap <- Sys.getenv("RETICULATE_REMAP_OUTPUT_STREAMS", unset = NA)
  if (!is.na(remap))
    force <- identical(remap, "1")

  # if stdout is NULL in Python we want to force remapping too.
  if (!force) {
    sys <- import("sys")
    force <- is.null(sys$stdout)
  }

  if (!force) return()

  set_output_streams(tty = interactive() || isatty(stdout()))
}

set_output_streams <- function(tty) {
  stream_context <- output_stream_context(tty)
  stream_context$`__enter__`()
}

set_knitr_python_stdout_hook <- function() {
  # if the env var is set to 0, we respect it and never remap. If the env var is 1
  # this means that remapping is already set and we don't need the hook anyway.
  if (!is.na(Sys.getenv("RETICULATE_REMAP_OUTPUT_STREAMS", unset = NA)))
    return()

  # we don't want to to force a knitr load namespace, so if it's already loaded
  # we set the knitr hook, otherwise we schedule an onLoad hook.
  if (isNamespaceLoaded("knitr")) {
    set_knitr_hook()

    # if knitr is already in progress here, this means that python was initialized
    # during a chunk execution. We have to force an instant remap as the hook won't
    # have a chance to run for that chunk.
    # In such cases `context.__enter__` is never called.
    if (isTRUE(getOption('knitr.in.progress'))) set_output_streams(tty = FALSE)
  } else {
    setHook(
      packageEvent("knitr", "onLoad"),
      function(...) {
        set_knitr_hook()
      }
    )
  }
}

set_knitr_hook <- function() {
  context <- output_stream_context(tty = FALSE)

  knitr::knit_hooks$set(include = function(before, options, envir) {
    if (!options$include) return()
    if (before) {
      context$`__enter__`()
    } else {
      context$`__exit__`()
    }
  })
}

output_stream_context <- function(tty) {
  output <- import("rpytools.output")
  output$RemapOutputStreams(
    write_stdout,
    write_stderr,
    tty = tty
  )
}

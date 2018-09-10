

remap_output_streams <- function() {
  output <- import("rpytools.output")

  # force remapping of output streams (required with Windows + Python 2
  # as otherwise output from Python is lost)
  force <- is_windows() && !is_python3()
  remap <- Sys.getenv("RETICULATE_REMAP_OUTPUT_STREAMS", unset = NA)
  if (!is.na(remap))
    force <- identical(remap, "1")

  output$remap_output_streams(
    write_stdout,
    write_stderr,
    tty = interactive() || isatty(stdout()),
    force = force
  )
}


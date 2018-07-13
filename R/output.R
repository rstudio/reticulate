

remap_output_streams <- function() {
  output <- import("rpytools.output")
  output$remap_output_streams(
    write_stdout,
    write_stderr,
    tty = interactive() || isatty(stdout()),
    force = is_windows() && !is_python3()
  )
}


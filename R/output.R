

write_stdout <- function(text) {
  write(text, stdout())
}

write_stderr <- function(text) {
  write(text, stderr())
}

remap_output_streams <- function() {
  output <- import("rpytools.output")
  output$remap_output_streams(write_stdout, write_stderr)
}
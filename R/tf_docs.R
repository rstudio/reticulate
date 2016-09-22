

view_docs <- function() {
  open <- ifelse(Sys.info()[["sysname"]] == "Darwin", "open", "xdg-open")
  system(paste(open, system.file("docs/index.html", package = "tensorflow")))
}

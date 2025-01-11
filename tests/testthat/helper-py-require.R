test_py_require_reset <- function() {
  .globals$python_requirements <- NULL
}

r_session <- function(expr, echo = TRUE, color = FALSE) {
  expr <- substitute(expr)
  if (is.call(expr) && identical(expr[[1]], quote(`{`))) {
    exprs <- as.list(expr)[-1]
  } else {
    exprs <- list(expr)
  }
  exprs <- unlist(lapply(exprs, deparse))
  writeLines(exprs, file <- tempfile(fileext = ".R"))
  on.exit(unlink(file), add = TRUE)

  result <- suppressWarnings(system2(
    R.home("bin/R"),
    c("--quiet", "--no-save", "--no-restore",
      if (!echo) "--no-echo",
      "-f", file
      ),
    stdout = TRUE, stderr = TRUE,
    env = c(if (isFALSE(color)) "NO_COLOR=1")
  ))
  class(result) <- "r_session_record"
  result
}

print.r_session_record <- function(record, echo = TRUE) {
  writeLines(record)
  status <- attr(record, "status", TRUE)
  cat(sep = "",
      "------- session end -------\n",
      "success: ", if (is.null(status)) "true" else "false", "\n",
      "exit_code: ", status %||% 0L, "\n")
}
registerS3method("print", "r_session_record", print.r_session_record,
                 envir = environment(print))

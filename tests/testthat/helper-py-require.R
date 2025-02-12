test_py_require_reset <- function() {
  .globals$python_requirements <- NULL
}

r_session <- function(exprs, echo = TRUE, color = FALSE,
                      attach_namespace = FALSE) {
  exprs <- substitute(exprs)
  if (!is.call(exprs))
    stop("exprs must be a call")

  exprs <- if (identical(exprs[[1]], quote(`{`)))
    as.list(exprs)[-1]
  else
    list(exprs)

  exprs <- unlist(c(
    if (attach_namespace)
      'attach(asNamespace("reticulate"), name = "namespace:reticulate", warn.conflicts = FALSE)',
    if (echo)
      "options(echo = TRUE)",
    lapply(exprs, deparse)
  ))

  writeLines(exprs, file <- tempfile(fileext = ".R"))
  on.exit(unlink(file), add = TRUE)

  result <- suppressWarnings(system2(
    R.home("bin/R"),
    c("--quiet", "--no-save", "--no-restore", "--no-echo", "-f", file),
    stdout = TRUE, stderr = TRUE,
    env = c(character(), if (isFALSE(color)) "NO_COLOR=1")
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


py_require_tested_packages <- function() {
  py_require(c(
    "docutils", "pandas", "scipy", "matplotlib", "ipython",
    "tabulate", "plotly", "psutil", "kaleido", "wrapt"
  ))
}

py_require_tested_packages()

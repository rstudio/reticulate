test_py_require_reset <- function() {
  .globals$python_requirements <- NULL
}

r_session <- function(exprs, echo = TRUE, color = FALSE,
                      attach_namespace = FALSE,
                      force_managed_python = TRUE) {
  withr::local_envvar(c(
    "VIRTUAL_ENV" = NA,
    "RETICULATE_PYTHON" = if (force_managed_python) "managed" else NA,
    "VIRTUAL_ENV_PROMPT" = NA,
    "RUST_LOG" = NA,
    "PYTHONPATH" = NA,
    "PYTHONIOENCODING" = "utf-8",
    if (isFALSE(color)) c("NO_COLOR" = "1")
  ))
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
    lapply(exprs, deparse, width.cutoff = 500L)
  ))

  writeLines(exprs, file <- tempfile(fileext = ".R"))
  on.exit(unlink(file), add = TRUE)

  result <- suppressWarnings(system2(
    R.home("bin/R"),
    c("--quiet", "--no-save", "--no-restore", "--no-echo", "-f", file),
    stdout = TRUE, stderr = TRUE
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


uninstall_system_uv <- function() {
  withr::local_envvar(c("NO_COLOR" = "1"))
  withr::local_path(path.expand("~/.local/bin"), action = "suffix")
  cache_dir <- system("uv cache dir", intern = TRUE) %error% NULL
  python_dir <- system("uv python dir", intern = TRUE) %error% NULL
  tool_dir <- system("uv tool dir", intern = TRUE) %error% NULL
  dirs <- c(cache_dir, python_dir, tool_dir)
  uv <- Sys.which("uv")
  uvx <- Sys.which("uvx")
  todelete <- c(dirs, uv, uvx)
  todelete <- todelete[nzchar(todelete) &
                         !is.na(todelete) & file.exists(todelete)]
  if (!length(todelete)) {
    message("nothing to delete")
    return()
  }
  todelete <- normalizePath(todelete)
  msg <- paste0("Delete?:\n", paste0("- ", todelete, collapse = "\n"), "\n")
  if (askYesNo(msg)) {
    unlink(todelete, recursive = TRUE, force = TRUE)
  }
}

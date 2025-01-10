test_py_require_reset <- function() {
  .globals$python_requirements <- NULL
}


eval_remote <- function(expr, echo = FALSE, error = TRUE) {
  expr <- substitute(expr)
  if(echo) {
    expr <- call("withAutoprint", expr)
  }
  func <- as.function.default(list(expr), envir = asNamespace("reticulate"))
  func <- rlang::zap_srcref(func)

  out <- tempfile("teststdout", fileext = ".out")
  err <- tempfile("teststderr", fileext = ".out")
  on.exit(unlink(c(stdout, stderr)))

  expect_error <- error
  encountered_error <- FALSE
  tryCatch(
    callr::r(
      func,
      spinner = FALSE,
      stdout = out,
      stderr = err,
      package = "reticulate",
      env = c(callr::rcmd_safe_env(), "NO_COLOR" = 1)
    ),

    error = function(e) {

      encountered_error <<- TRUE

      if(file.exists(out)) writeLines(readLines(out))

      # evaluate::evaluate(), testthat::expect_snapshot(), and friends
      # do not capture stderr.
      # https://github.com/r-lib/testthat/issues/1741
      # https://github.com/r-lib/evaluate/issues/121
      # to play nice with expect_snapshot(), options are to
      #  - print via message()
      #  - print to stdout() using writeLines(foo, stderr())
      if(file.exists(err)) message(paste0(readLines(err), collapse = "\n"))

      if (!expect_error)
        stop(e)
    }
  )
  if (expect_error && !encountered_error) {
    stop("expression did not error")
  }
}

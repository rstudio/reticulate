
run <- function() {

  on_cran <- !isTRUE(as.logical(Sys.getenv("NOT_CRAN", "false")))
  if (on_cran) {
    message("env var 'NOT_CRAN=true' not defined; Skipping tests on CRAN")
    return()
  }

  if (!requireNamespace("testthat", quietly = TRUE)) {
    message("'testthat' package not available; tests cannot be run")
    return()
  }

  # options(error = traceback)
  # if (requireNamespace("rlang", quietly = TRUE))
  #   options(error = rlang::trace_back)

  library(testthat)
  library(reticulate)

  test_check("reticulate")

}

run()

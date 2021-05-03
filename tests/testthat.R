
run <- function() {
  
  if (!requireNamespace("testthat", quietly = TRUE)) {
    message("'testthat' package not available; tests cannot be run")
    return()
  }
  
  library(testthat)
  library(reticulate)
  
  test_check("reticulate")
  
}

run()

Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python3")

library(testthat)
library(reticulate)

test_check("reticulate")

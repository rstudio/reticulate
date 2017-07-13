library(testthat)
library(reticulate)

py_discover_config("numpy")
py_config()

test_check("reticulate")

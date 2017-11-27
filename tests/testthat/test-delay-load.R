context("delay-load")

source("utils.R")

library(callr)

test_that("imoprted module can be customized via delay_load", {
  skip_if_no_python()
  expect_true(r(function() {
    sys <- reticulate::import("invalid_module_name", delay_load = list(
      get_module = function() {
        "sys"
      }
    ))
    is.character(sys$copyright)
  }))
})


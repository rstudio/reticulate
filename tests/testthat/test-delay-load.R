context("delay-load")

library(callr)

test_that("imoprted module can be customized via delay_load", {
  skip_if_no_python()
  skip_on_travis()
  expect_true(r(function() {
    sys <- reticulate::import("invalid_module_name", delay_load = list(
      get_module = function() {
        "sys"
      }
    ))
    is.character(sys$byteorder)
  }))
})


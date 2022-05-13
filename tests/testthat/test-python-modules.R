context("modules")

test_that("modules can be imported, printed with 'as'", {
  skip_on_cran()
  # previously failed when attempting to print the module
  # https://github.com/rstudio/reticulate/issues/631
  module <- import("time", as = "t")
  expect_output(print(module), "Module(time)", fixed = TRUE)
  expect_true(inherits(module, "python.builtin.module"))
})

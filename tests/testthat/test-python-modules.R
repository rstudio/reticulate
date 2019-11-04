context("modules")

test_that("modules can be imported, printed with 'as'", {
  # previously failed when attempting to print the module
  # https://github.com/rstudio/reticulate/issues/631
  module <- import("time", as = "t")
  expect_equal(py_str(module), "Module(time)")
  expect_true(inherits(module, "python.builtin.module"))
})

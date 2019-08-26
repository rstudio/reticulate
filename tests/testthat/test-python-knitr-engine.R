context("knitr")

test_that("An R Markdown document can be rendered using reticulate", {
  skip_on_cran()
  skip_on_os("windows")
  skip_if_no_python()
  skip_if_not_installed("rmarkdown")

  modules <- c("numpy", "matplotlib")
  for (module in modules) {
    if (!py_module_available(module)) {
      fmt <- "module '%s' not available; skipping"
      skip(sprintf(fmt, module))
    }
  }

  owd <- setwd("resources")
  status <- rmarkdown::render("eng-reticulate-example.Rmd", quiet = TRUE)
  setwd(owd)

  expect_true(file.exists(status), "example.Rmd rendered successfully")
})



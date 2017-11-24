context("knitr")

test_that("An R Markdown document can be rendered using reticulate", {
  
  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  
  modules <- c("numpy", "matplotlib")
  for (module in modules) {
    if (!py_module_available(module)) {
      fmt <- "module '%s' not available; skipping"
      skip(sprintf(fmt, module))
    }
  }
  
  status <- withr::with_dir("resources", {
    rmarkdown::render("example.Rmd", quiet = TRUE)
  })
  expect_true(file.exists(status), "example.Rmd rendered successfully")
})

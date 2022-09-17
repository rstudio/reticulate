context("knitr-engine")

test_that("An R Markdown document can be rendered using reticulate", {

  skip_on_cran()
  skip_on_os("windows")
  skip_if_not_installed("rmarkdown")
  skip_if(py_version() < "3") # plotly _repr_html_ test fails in py2


  modules <- c("numpy", "matplotlib", "pandas", "plotly",
               "tabulate", "IPython")
  for (module in modules) {
    if (!py_module_available(module)) {
      fmt <- "module '%s' not available; skipping"
      skip(sprintf(fmt, module))
    }
  }

  output <- withr::with_dir("resources", {
    rmarkdown::render("eng-reticulate-example.Rmd", quiet = TRUE)
  })

  expect_true(file.exists(output), "example.Rmd rendered successfully")
})

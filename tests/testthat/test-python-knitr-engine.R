context("knitr")

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

  owd <- setwd("resources")
  status <- rmarkdown::render("eng-reticulate-example.Rmd", quiet = TRUE)
  setwd(owd)

  expect_true(file.exists(status), "example.Rmd rendered successfully")
})




test_that("In Rmd chunks, comments and output attach to code correctly", {

  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  skip_if(py_version() < "3.8") # end_lineno attr added in 3.8

  local_edition(3) # needed for expect_snapshot_file()
  # TODO: update the full testsuite to testthat edition 3

  print(getwd())
  owd <- setwd(test_path("resources"))
  on.exit(owd, add = TRUE)
  rmarkdown::render("test-chunking.Rmd")

  expect_snapshot_file("test-chunking.md")

})

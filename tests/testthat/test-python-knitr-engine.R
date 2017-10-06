context("knitr")

test_that("An R Markdown document can be rendered using reticulate", {
  skip_if_not_installed("rmarkdown")
  status <- withr::with_dir("resources", {
    rmarkdown::render("example.Rmd", quiet = TRUE)
  })
  expect_true(file.exists(status), "example.Rmd rendered successfully")
})

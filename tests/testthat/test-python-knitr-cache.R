context("knitr-cache")

test_that("An R Markdown document using reticulate can be rendered with cache feature", {

  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  skip_if_not(cache_eng_python$available(knitr::opts_chunk$get()))

  flag_file <- "py_chunk_executed"
  rmd_prefix <- "eng-reticulate-cache"
  rmd_file <- paste(rmd_prefix, "Rmd", sep=".")

  withr::with_dir("resources", local({
    withr::defer({
      unlink(flag_file)
      unlink(paste(rmd_prefix, "cache", sep="_"), recursive = TRUE)
    })

    # cache file is created
    output <- rmarkdown::render(rmd_file, quiet = TRUE)
    expect_true(file.exists(flag_file))
    expect_true(file.exists(output))
    unlink(c(output, flag_file))

    # cached results should be used
    output <- rmarkdown::render(rmd_file, quiet = TRUE)
    expect_false(file.exists(flag_file))
    expect_true(file.exists(output))
  }))
})

context("knitr-cache")

test_that("An R Markdown document using reticulate can be rendered with cache feature", {

  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  skip_if_not(cache_eng_python$available(knitr::opts_chunk$get()))

  flag_file <- "py_chunk_executed"
  rmd_prefix <- "eng-reticulate-cache"
  rmd_file <- paste(rmd_prefix, "Rmd", sep=".")
  cache_path <- paste(rmd_prefix, "cache", sep="_")

  withr::with_dir("resources", local({
    withr::defer({
      unlink(flag_file)
      unlink(cache_path, recursive = TRUE)
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

    # the 'spam' variable should not be cached in the 'cache-vars' block
    main <- import_main()
    dill <- import("dill")
    py_del_attr(main, "spam")
    session_file <- Sys.glob(paste0(cache_path, "/*/cache-vars_*.pkl"))
    dill$load_module(session_file)
    expect_false("spam" %in% names(main))
  }))
})

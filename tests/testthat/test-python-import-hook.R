context("imports")

test_that("The reticulate import hook handles recursive imports", {

  skip_if_no_matplotlib()

  R <- file.path(R.home("bin"), "R")
  script <- "resources/import-test.R"
  args <- c("--no-save", "--no-restore", "-s", "-f", shQuote(script))
  output <- system2(R, args, stdout = TRUE, stderr = TRUE)

  pattern <- "Loaded module '(.*)'"
  modules <- gsub(pattern, "\\1", output)
  expect_true("matplotlib.pyplot" %in% modules)

})

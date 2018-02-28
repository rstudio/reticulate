context("knitr-cache")

test_that("An R Markdown document can be rendered with cache using reticulate", {
  
  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  skip_if_not_installed("callr")
  
  unlink("resources/eng-reticulate-cache-test_cache/", recursive = TRUE)
  
  path <- callr::r(
    function() {
      rmarkdown::render("resources/eng-reticulate-cache-test.Rmd", quiet = TRUE, envir = new.env())
    })
  expect_true(file.exists(path))
  on.exit(unlink(path), add = TRUE)
})

test_that("An R Markdown document builds if a cache is modified", {
  
  skip_on_cran()
  skip_if_not_installed("rmarkdown")
  skip_if_not_installed("callr")
  
  old_var <- "1"
  new_var <- "0"
  mutate_chunk <- function(x) {
    print_line <- 19
    file_text <- readLines("resources/eng-reticulate-cache-test.Rmd")
    file_text[print_line] <- paste("print(x + ", x, ")", sep = "")
    writeLines(file_text, "resources/eng-reticulate-cache-test.Rmd")
  }
  mutate_chunk(old_var)
  mutate_chunk(new_var)
  path <- callr::r(
    function() {
      rmarkdown::render("resources/eng-reticulate-cache-test.Rmd", quiet = TRUE, envir = new.env())
    })
  mutate_chunk(old_var)
  expect_true(file.exists(path))
  on.exit(unlink(path), add = TRUE)
  on.exit(unlink("resources/eng-reticulate-cache-test_cache/", recursive = TRUE), add = TRUE)
})




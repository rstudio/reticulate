context("examples")

source("utils.R")

# some helpers
run_example <- function(example) {
  env <- new.env()
  capture.output({
    example_path <- system.file("examples", example, package = "rpy")
    old_wd <- setwd(dirname(example_path))
    on.exit(setwd(old_wd), add = TRUE)
    source(basename(example_path), local = env)
  }, type = "output")
  rm(list = ls(env), envir = env)
  gc()
}

examples <- c()

for (example in examples) {
  test_that(paste(example, "example runs successfully"), {
    skip_if_no_python()
    expect_error(run_example(example), NA)
  })
}


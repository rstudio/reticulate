context("examples")

# some helpers
run_example <- function(example) {
  env <- new.env()
  capture.output({
    source(system.file("examples", example, package = "tensorflow"), local = env)
  }, type = "output")
  rm(list = ls(env), envir = env)
  gc()
}

examples <- c("hello.R", "introduction.R")

for (example in examples) {
  test_that(paste(example, "example runs successfully"), {
    expect_error(run_example(example), NA)
  })
}


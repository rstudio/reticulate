context("examples")

# some helpers
run_example <- function(example) {
  env <- new.env()
  source(system.file("examples", example, package = "tensorflow"), local = env)
  rm(list = ls(env), envir = env)
  gc()
}

examples <- c("hello.R")

for (example in examples) {
  test_that(paste(example, "example runs successfully"), {
    expect_error(run_example(example), NA)
  })
}


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

examples <- c("hello.R", "introduction.R", "mnist_softmax.R")

for (example in examples) {
  test_that(paste(example, "example runs successfully"), {
    skip_on_cran()
    skip_on_os("mac")
    expect_error(run_example(example), NA)
  })
}


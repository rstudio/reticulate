context("examples")

source("utils.R")

# some helpers
run_example <- function(example) {
  env <- new.env()
  capture.output({
    source(system.file("examples", example, package = "tensorflow"), local = env)
  }, type = "output")
  rm(list = ls(env), envir = env)
  gc()
}

examples <- c("hello.R", "introduction.R", "mnist/mnist_softmax.R")

for (example in examples) {
  test_that(paste(example, "example runs successfully"), {
    skip_if_no_tensorflow()
    expect_error(run_example(example), NA)
  })
}


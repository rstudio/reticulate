
test_that("Running Python scripts can be interrupted", {
  
  skip_on_cran()
  
  # import time module
  time <- import("time", convert = TRUE)
  
  # interrupt this process shortly
  system(paste("sleep 1 && kill -s INT", Sys.getpid()), wait = FALSE)
  
  # tell Python to sleep
  before <- Sys.time()
  interrupted <- tryCatch(time$sleep(5), interrupt = identity)
  after <- Sys.time()
  
  # check that we caught an interrupt
  expect_s3_class(interrupted, "interrupt")
  
  # check that we took a small amount of time
  diff <- difftime(after, before, units = "secs")
  expect_true(diff < 2)
  
})

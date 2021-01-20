
test_that("Running Python scripts can be interrupted", {
  
  skip_on_os("windows")
  
  time <- import("time")
  
  # interrupt this process in a couple seconds
  system(paste("sleep 1 && kill -s INT", Sys.getpid()), wait = FALSE)
  
  # tell Python to sleep
  before <- Sys.time()
  time$sleep(10)
  after <- Sys.time()
  
  # check that we took a small amount of time
  diff <- difftime(after, before, units = "secs")
  expect_true(diff < 2)
  
})

context("output")

capture_test_output <- function(type) {
  py_capture_output(type = type, {
    if ("stdout" %in% type)
      sys$stdout$write("out");
    if ("stderr" %in% type)
    sys$stderr$write("err");
  })
}

test_that("Python streams can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = c("stdout", "stderr")) ,"out\nerr\n")
})

test_that("Python stdout stream can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = "stdout") , "out\n")
})

test_that("Python stderr stream can be captured", {
  skip_if_no_python()
  expect_equal(capture_test_output(type = "stderr") , "err\n")
})

test_that("Python loggers work with py_capture_output", {
  skip_if_no_python()
  
  output <- py_capture_output({
    logging <- import("logging")
    l <- logging$getLogger("test.logger")
    l$addHandler(logging$StreamHandler())
    l$setLevel("INFO")
    l$info("info")
  })
  
  expect_equal(output, "info\n\n")
  
  l <- logging$getLogger("test.logger2")
  l$addHandler(logging$StreamHandler())
  l$setLevel("INFO")
  output <- py_capture_output(l$info("info"))
  
  expect_equal(output, "info\n\n")
  
})

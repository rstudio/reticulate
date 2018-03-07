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



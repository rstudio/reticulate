context("run")

test_that("Python code can be run as strings", {

  # runs code in main module
  result <- py_run_string("x = 1")
  expect_equal(result$x, 1L)

  main <- import_main(convert = TRUE)
  expect_equal(main$x, 1L)

  # runs code in local dictionary
  result <- py_run_string("x = 42", local = TRUE)
  expect_true(result$x == 42L)
  expect_true(main$x == 1L)

})



test_that("Python files can be run", {

  file <- tempfile(fileext = ".py")
  writeLines("file = __file__", file)

  out <- py_run_file(file, local = TRUE)
  expect_s3_class(out, "python.builtin.dict")
  expect_equal(file, out$file)
  expect_false("__name__" %in% names(out))


  out <- py_run_file(file, local = FALSE)
  expect_s3_class(out, "python.builtin.dict")
  expect_identical(get("pyobj", out),
                   get("pyobj", py_get_attr(import_main(), "__dict__")))
  expect_equal(file, out$file)
  expect_false("__name__" %in% names(out))

  py_run_string("del file") # cleanup after test
})

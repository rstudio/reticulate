context("source")

test_that("Python scripts can be sourced from local file", {
  skip_if_no_python()
  source_python(test_path('script.py'))
  expect_equal(add(2, 4), 6)
})

test_that("Python scripts can be sourced from a URL", {
  skip_if_no_python()
  source_python('https://raw.githubusercontent.com/rstudio/reticulate/main/tests/testthat/script.py')
  expect_equal(add(2, 4), 6)
})

test_that("source_python assigns into the requested environment", {
  skip_if_no_python()
  env <- new.env(parent = emptyenv())
  source_python(test_path('script.py'), envir = env)
  expect_equal(env$add(2, 4), 6)
})

test_that("source_python respects the convert argument", {
  skip_if_no_python()
  source_python(test_path('script.py'), convert = FALSE)
  expect_s3_class(add(2, 4), 'python.builtin.object')
})

test_that("python functions can call each other", {
  skip_if_no_python()
  source_python(test_path('script.py'))
  expect_equal(secret(), 42)
  expect_equal(api(), 42)
})

test_that("source_python() overlays in the main module", {
  skip_if_no_python()
  source_python(test_path('script.py'))
  main <- import_main()
  expect_equal(main$value, 42)
})


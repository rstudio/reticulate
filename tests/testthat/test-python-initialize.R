context("py_initialize")

# The package's test_that() skips tests when Python fails to initialize.
testthat::test_that("Python initializes with the same pointer width as R", {
  testthat::skip_if_not(file.exists(Sys.getenv("RETICULATE_PYTHON")),
                        "requires an explicitly configured interpreter")
  size <- callr::r(function() reticulate::py_eval("__import__('struct').calcsize('P')"))
  expect_identical(size, .Machine$sizeof.pointer)
})

test_that("Python 2 is rejected before initialization changes the environment", {
  result <- callr::r(function() {
    testthat::local_mocked_bindings(
      py_discover_config = function(...) list(version = numeric_version("2.7")),
      .package = "reticulate"
    )
    variables <- c("VIRTUAL_ENV", "R_SESSION_INITIALIZED", "PATH", "PYTHONPATH")
    before <- Sys.getenv(variables, unset = NA_character_)
    error <- tryCatch(reticulate::py_config(), error = conditionMessage)

    list(error = error,
         initialized = reticulate::py_available(initialize = FALSE),
         before = before,
         after = Sys.getenv(variables, unset = NA_character_))
  })

  expect_identical(result$error, "Python 3 or newer is required; found Python 2.7.")
  expect_false(result$initialized)
  expect_identical(result$after, result$before)
})

test_that("sys.executable points to the correct python", {


  py_exe_stand_alone <-  system2(py_exe(), c("-c", shQuote("import sys; print(sys.executable)")), stdout = TRUE)
  py_exe_embedded <- import("sys")$executable

  if (is_windows())
    py_exe_embedded <- utils::shortPathName(py_exe_embedded)

   expect_identical(py_exe_stand_alone, py_exe_embedded)

})


test_that("__main__ initialized with only 'r'", {

  expect_identical(
    callr::r(function() names(reticulate::import_main())),
    "r")
})

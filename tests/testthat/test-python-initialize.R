context("py_initialize")

in_fresh_r <- function(code) {
  source_path <- if (pkgload::is_dev_package("reticulate"))
    getNamespaceInfo("reticulate", "path")

  callr::r(function(path, code) {
    if (!is.null(path))
      pkgload::load_all(path, export_all = FALSE, helpers = FALSE)
    eval(code, globalenv())
  }, args = list(source_path, substitute(code)))
}

# The package's test_that() skips tests when Python fails to initialize.
testthat::test_that("Python initializes with the same pointer width as R", {
  testthat::skip_if_not(file.exists(Sys.getenv("RETICULATE_PYTHON")),
                        "requires an explicitly configured interpreter")
  size <- py_eval("__import__('struct').calcsize('P')")
  expect_identical(size, .Machine$sizeof.pointer)
})

test_that("Python 2 is rejected before initialization changes the environment", {
  result <- in_fresh_r({
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
    in_fresh_r(names(reticulate::import_main())),
    "r")
})

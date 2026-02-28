context("virtual environments")

test_that("reticulate can bind to virtual environments created with venv", {

  skip_if_no_python()
  skip_on_cran()
  skip_on_os("windows")

  Sys.unsetenv("RETICULATE_PYTHON")
  Sys.unsetenv("RETICULATE_PYTHON_ENV")

  # ensure cacert.pem goes to right folder
  withr::local_envvar(TMPDIR = tempdir())
  # this test only verifies venv activation mechanics; package requirement
  # warnings are covered in test-py_require.R snapshots.
  withr::local_envvar(RETICULATE_CHECK_REQUIRED_PACKAGES = "false")

  # find Python 3 binary for testing
  python3 <- Sys.which("python3")
  if (!nzchar(python3))
    skip("test requires Python 3 binary with venv module")

  # create a virtual environment in the tempdir
  venv <- tempfile("python-3-venv")
  status <- tryCatch(system(paste(python3, "-m venv", venv)), error = identity)
  if (inherits(status, "error"))
    skip("test requires Python 3 binary with venv module")

  on.exit(unlink(venv, recursive = TRUE), add = TRUE)
  venv <- normalizePath(venv, mustWork = TRUE)

  # try running reticulate and binding to that virtual environment
  R <- file.path(R.home("bin"), "R")
  script <- normalizePath("resources/venv-activate.R")
  args <- c("--vanilla", "--slave", "-f", shQuote(script), "--args", shQuote(venv))
  output <- system2(R, args, stdout = TRUE)

  # test that we're using the site-packages dir in the venv
  site <- grep("site-packages", output, fixed = TRUE, value = TRUE)
  expect_true(any(grepl(venv, site)))

})

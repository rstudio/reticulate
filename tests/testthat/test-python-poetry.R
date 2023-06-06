context("poetry")

test_that("reticulate uses the Poetry-configured version of Python", {

  if (!nzchar(Sys.which("poetry")))
    skip("poetry is not installed")

  # unset RETICULATE_PYTHON in this scope
  withr::local_envvar(RETICULATE_PYTHON = NULL)

  # move to temporary directory
  project <- tempfile("poetry-")
  dir.create(project)
  on.exit(unlink(project), add = TRUE)

  owd <- setwd(project)
  on.exit(setwd(owd), add = TRUE)

  # initialize project
  system("poetry new .", ignore.stdout = TRUE, ignore.stderr = TRUE)
  expect_true(file.exists("pyproject.toml"))

  # remove dependency on pytest (poetry solver barfs?)
  contents <- readLines("pyproject.toml")
  contents <- grep("^pytest", contents, invert = TRUE, value = TRUE)
  writeLines(contents, con = "pyproject.toml")

  # try running python (force initialization of virtualenv)
  system("poetry run python --version")

  # ask for virtualenv path
  envpath <- system("poetry env info --path", intern = TRUE)
  expected <- virtualenv_python(envpath)

  # try running reticulate in child process
  fmt <- "R -s -e '%s'"
  cmd <- sprintf(fmt, "writeLines(reticulate::py_config()$python)")
  actual <- system(cmd, intern = TRUE)

  expect_equal(
    normalizePath(expected),
    normalizePath(actual)
  )

})

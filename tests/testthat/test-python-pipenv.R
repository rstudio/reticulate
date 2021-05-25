context("pipenv")

test_that("reticulate uses the pipenv-configured version of Python", {
  
  if (!nzchar(Sys.which("pipenv")))
    skip("pipenv is not installed")
  
  # move to temporary directory
  project <- tempfile("pipenv-")
  dir.create(project)
  on.exit(unlink(project), add = TRUE)
  
  owd <- setwd(project)
  on.exit(setwd(owd), add = TRUE)
  
  # initialize a pipenv project
  system("pipenv install", ignore.stdout = TRUE, ignore.stderr = TRUE)
  
  # ask for virtualenv path
  expected <- system("pipenv --py", intern = TRUE)
  
  # try running reticulate in child process
  fmt <- "R --vanilla -s -e '%s'"
  cmd <- sprintf(fmt, "writeLines(reticulate::py_config()$python)")
  actual <- system(cmd, intern = TRUE)
  
  expect_equal(
    normalizePath(expected),
    normalizePath(actual)
  )
  
})

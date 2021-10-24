context("poetry")

test_that("reticulate uses the poetry-configured version of Python", {
  
  if (!nzchar(Sys.which("poetry")))
    skip("poetry is not installed")
  
  # move to temporary directory
  project <- tempfile("poetry-")
  dir.create(project)
  on.exit(unlink(project), add = TRUE)
  
  owd <- setwd(project)
  on.exit(setwd(owd), add = TRUE)
  
  # initialize a poetry project
  system("poetry init --no-interaction --quiet", ignore.stdout = TRUE, ignore.stderr = TRUE)
  # make the venv in the project folder to ensure it is cleaned up on exit
  cat(c("[virtualenvs]", "in-project = true"), file = "poetry.toml", sep = "\n")
  system("poetry env use python --quiet", ignore.stdout = TRUE, ignore.stderr = TRUE)
  
  # ask for virtualenv path
  expected <- system("poetry env info --path", intern = TRUE)
  expected <- paste0(expected, "/bin/python")
  
  # try running reticulate in child process
  fmt <- "R --vanilla -s -e '%s'"
  cmd <- sprintf(fmt, "writeLines(reticulate::py_config()$python)")
  actual <- system(cmd, intern = TRUE)
  
  expect_equal(
    normalizePath(expected),
    normalizePath(actual)
  )
  
})

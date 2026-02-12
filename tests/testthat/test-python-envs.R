context("envs")

python_is_free_threaded_via_vv <- function(python) {
  output <- tryCatch(
    system2(python, c("-E", "-VV"), stdout = TRUE, stderr = TRUE),
    error = identity
  )
  if (inherits(output, "error"))
    return(FALSE)

  status <- attr(output, "status") %||% 0L
  if (status != 0L || !length(output))
    return(FALSE)

  any(grepl("free-thread(?:ed|ing) build", output, ignore.case = TRUE))
}

test_that("conda utility functions work as expected", {
  # TODO: reenable these tests
  skip_if_no_test_environments()

  binary <- conda_binary()
  expect_type(binary, "character")
  expect_length(binary, 1)

  conda_remove('reticulate-testthat')
  conda_create('reticulate-testthat')
  expect_true('reticulate-testthat' %in% conda_list()$name)

  conda_install('reticulate-testthat', 'Pillow')
  conda_remove('reticulate-testthat', 'Pillow')

  conda_remove('reticulate-testthat')
  expect_false('reticulate-testthat' %in% conda_list()$name)

  conda_create('reticulate-testthat', forge = TRUE)
  expect_true(all(grepl("conda-forge", conda_list_packages("reticulate-testthat")$channel)))
  conda_remove('reticulate-testthat')

  conda_create('reticulate-testthat', channel = c("anaconda"))
  expect_true(all(grepl("anaconda", conda_list_packages("reticulate-testthat")$channel)))
  conda_remove('reticulate-testthat')

})

test_that("virtualenv utility functions work as expected", {
  skip_if_no_test_environments()

  expect_error(
    virtualenv_remove('reticulate-testthat', confirm = FALSE),
    'Virtual environment \'reticulate-testthat\' does not exist.'
  )

  virtualenv_create('reticulate-testthat')
  virtualenv_remove('reticulate-testthat', confirm = FALSE)

  virtualenv_install('reticulate-testthat', 'Pillow')
  virtualenv_install('reticulate-testthat', 'Pillow', ignore_installed = TRUE)

  expect_true('reticulate-testthat' %in% virtualenv_list())

  virtualenv_remove('reticulate-testthat', confirm = FALSE)

  expect_false('reticulate-testthat' %in% virtualenv_list())

})

test_that("Python version checker support a 'x.x.*' pattern", {
  check <- as_version_constraint_checkers("==3.12.*")
  expect_false(check[[1]]("3.9"))
  expect_true(check[[1]]("3.12"))
})

test_that("Python version checker returns expected error message", {
  check <- as_version_constraint_checkers("==3.12.-")
  expect_error(check[[1]]("3.12"), "Version `==3.12.-` is not valid.")
})

test_that("Python version checker does not support pattern 'x.*.x'", {
  check <- as_version_constraint_checkers(">=3.*.11")
  expect_false(check[[1]]("3.12"))
  expect_false(check[[1]]("4.1"))
})

test_that("python_is_free_threaded matches interpreter config", {
  skip_if_no_python()

  python <- Sys.which("python3")
  if (!nzchar(python))
    python <- Sys.which("python")
  if (!nzchar(python))
    skip("No python interpreter available for testing")

  expect_identical(
    python_is_free_threaded(python),
    python_is_free_threaded_via_vv(python)
  )
})

test_that("virtualenv_starter excludes free-threaded Python builds", {
  skip_if_no_python()

  starters <- virtualenv_starter(all = TRUE)
  if (!nrow(starters))
    skip("No virtualenv starters found")

  is_free_threaded <- vapply(
    starters$path,
    python_is_free_threaded_via_vv,
    logical(1),
    USE.NAMES = FALSE
  )
  expect_false(any(is_free_threaded))
})

test_that("virtualenv_starter rejects explicitly requested free-threaded Python", {
  candidates <- c(
    Sys.glob("~/.pyenv/versions/*t*/bin/python*"),
    Sys.glob("~/.pyenv/versions/*t*/python*.exe")
  )
  candidates <- candidates[grep("^python[0-9.]*(\\.exe)?$", basename(candidates))]
  candidates <- candidates[utils::file_test("-x", candidates)]
  candidates <- candidates[utils::file_test("-f", candidates)]
  candidates <- unique(normalizePath(candidates, winslash = "/", mustWork = FALSE))
  candidates <- candidates[vapply(
    candidates, python_is_free_threaded_via_vv, logical(1), USE.NAMES = FALSE
  )]

  if (!length(candidates))
    skip("No free-threaded Python installations found")

  expect_null(virtualenv_starter(candidates[[1L]]))
})

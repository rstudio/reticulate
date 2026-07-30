context("pipenv")

testthat::test_that("pipenv discovery uses the located Pipfile directory", {

  skip_on_cran()

  python <- Sys.which("python")
  if (!nzchar(python))
    python <- Sys.which("python3")
  if (!nzchar(python))
    skip("python is not installed")

  result <- callr::r(
    function(python) {

      project <- tempfile("pipenv-")
      dir.create(project)
      on.exit(unlink(project, recursive = TRUE), add = TRUE)

      pipfile <- file.path(project, "Pipfile")
      file.create(pipfile)

      envpath <- file.path(project, ".venv")
      status <- system2(
        python,
        c("-m", "venv", "--without-pip", shQuote(envpath)),
        stdout = FALSE,
        stderr = FALSE
      )
      stopifnot(status == 0L)

      bindir <- file.path(project, "bin")
      dir.create(bindir)

      if (.Platform$OS.type == "windows") {
        pipenv <- file.path(bindir, "pipenv.bat")
        writeLines(c("@echo off", paste("echo", envpath)), pipenv)
      } else {
        pipenv <- file.path(bindir, "pipenv")
        command <- sprintf("printf '%%s\\n' %s", shQuote(envpath))
        writeLines(c("#!/bin/sh", command), pipenv)
        Sys.chmod(pipenv, "0755")
      }

      withr::local_path(bindir)
      Sys.unsetenv(c(
        "PYTHON_SESSION_INITIALIZED",
        "RETICULATE_PYTHON",
        "RETICULATE_PYTHON_ENV",
        "RETICULATE_PYTHON_FALLBACK",
        "VIRTUAL_ENV"
      ))
      Sys.setenv(RETICULATE_USE_MANAGED_VENV = "false")

      config <- testthat::with_mocked_bindings(
        reticulate::py_discover_config(),
        here = function(...) {
          path <- list(...)
          if (identical(path, list("Pipfile")))
            return(pipfile)
          stop("No root directory found", call. = FALSE)
        },
        .package = "here"
      )

      list(
        actual = config$python,
        expected = reticulate::virtualenv_python(envpath),
        forced = config$forced
      )

    },
    args = list(python)
  )

  expect_equal(result$actual, result$expected)
  expect_equal(result$forced, "Pipfile")

})

test_that("reticulate uses the pipenv-configured version of Python", {

  skip_on_cran()
  if (!nzchar(Sys.which("pipenv")))
    skip("pipenv is not installed")

  # use R session directory for tempdir
  withr::local_envvar(TMPDIR = tempdir())

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

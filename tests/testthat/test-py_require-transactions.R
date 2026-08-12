test_that("py_require() removes exact package requirements", {
  session <- r_session(echo = FALSE, {
    library(reticulate)
    py_require(c("numpy", "numpy==2", "pandas"), action = "set")
    py_require(c("numpy", "not-requested"), action = "remove")

    stopifnot(identical(py_require()$packages, c("numpy==2", "pandas")))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("py_require() sets packages and Python versions independently", {
  session <- r_session(echo = FALSE, {
    library(reticulate)
    py_require(
      packages = c("numpy", "pandas"),
      python_version = c(">=3.10", "<3.14"),
      action = "set"
    )
    py_require(packages = "numpy", action = "set")

    requirements <- py_require()
    stopifnot(identical(requirements$packages, "numpy"))
    stopifnot(identical(
      requirements$python_version,
      c(">=3.10", "<3.14")
    ))

    py_require(python_version = "3.12", action = "set")

    requirements <- py_require()
    stopifnot(identical(requirements$packages, "numpy"))
    stopifnot(identical(requirements$python_version, "3.12"))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("py_require() clears exclude_newer with its documented sentinels", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    py_require(exclude_newer = "2020-01-01")
    py_require(exclude_newer = NA, action = "remove")
    stopifnot(is.null(py_require()$exclude_newer))
    stopifnot(identical(
      names(py_require()),
      c("packages", "python_version", "exclude_newer")
    ))

    py_require(exclude_newer = "2020-01-01")
    py_require(exclude_newer = "", action = "remove")
    stopifnot(is.null(py_require()$exclude_newer))
    stopifnot(identical(
      names(py_require()),
      c("packages", "python_version", "exclude_newer")
    ))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("querying requirements does not invoke uv", {
  session <- r_session(echo = FALSE, {
    Sys.setenv(RETICULATE_UV = file.path(tempdir(), "uv-that-must-not-run"))
    library(reticulate)

    requirements <- py_require()
    stopifnot(identical(
      names(requirements),
      c("packages", "python_version", "exclude_newer")
    ))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printing names the default Python and source of requirements", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")
    package_py_require("example-package")

    printed <- capture.output(print(py_require()))
    stopifnot(any(grepl("Will default to", printed, fixed = TRUE)))
    stopifnot(any(grepl(
      "Python requirements declared by R packages:",
      printed,
      fixed = TRUE
    )))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printed request sources are part of the returned snapshot", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")

    package_py_require("first-package")
    requirements <- py_require()
    package_py_require("second-package")
    before <- capture.output(print(requirements))
    after <- capture.output(print(py_require()))

    stopifnot(any(grepl("first-package", before, fixed = TRUE)))
    stopifnot(!any(grepl("second-package", before, fixed = TRUE)))
    stopifnot(any(grepl("second-package", after, fixed = TRUE)))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("failed package late additions do not change requirements", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    py_require(python_version = ">=3.8")
    import("sys")

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")

    before <- py_require()
    before_print <- capture.output(print(before))
    package_py_require("not a valid requirement")
    after <- py_require()

    stopifnot(length(warnings()) > 0L)
    stopifnot(identical(
      before[c("packages", "python_version", "exclude_newer")],
      after[c("packages", "python_version", "exclude_newer")]
    ))
    stopifnot(identical(before_print, capture.output(print(after))))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

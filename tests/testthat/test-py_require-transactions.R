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

    py_require(exclude_newer = "2020-01-01")
    py_require(exclude_newer = "", action = "remove")
    stopifnot(is.null(py_require()$exclude_newer))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printing requirements has a base fallback", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")
    package_py_require("example-package", python_version = ">=3.10")

    printed <- testthat::with_mocked_bindings(
      capture.output(print(py_require())),
      requireNamespace = function(...) FALSE,
      .package = "base"
    )
    stopifnot(
      grepl("^=+ Python requirements =+$", printed[[1L]]),
      any(grepl("^-- Current requirements -+$", printed)),
      any(grepl(
        "^-- Python requirement requests \\(in order\\) -+$",
        printed
      )),
      any(grepl("R package stats", printed, fixed = TRUE)),
      any(grepl("^    Action:   add$", printed))
    )
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printing initialized requirements uses the active Python", {
  session <- r_session(echo = FALSE, {
    library(reticulate)
    import("sys")

    version <- as.character(py_version(patch = TRUE))
    Sys.setenv(RETICULATE_UV = file.path(tempdir(), "uv-that-must-not-run"))
    printed <- capture.output(print(py_require()))

    stopifnot(any(grepl(
      sprintf("Defaulted to '%s'", version),
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

    package_py_require("first-package", python_version = ">=3.10")
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

test_that("diagnostic history retains committed requests in order", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    py_require("global-package")

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)

    environment(package_py_require) <- asNamespace("stats")
    package_py_require("diagnostic-package")

    environment(package_py_require) <- asNamespace("graphics")
    package_py_require("diagnostic-package", action = "remove")

    requirements <- py_require()
    history <- requirements$history
    events <- tail(history, 3L)

    stopifnot(identical(
      names(requirements),
      c("python_version", "packages", "exclude_newer", "history")
    ))
    stopifnot(!"diagnostic-package" %in% requirements$packages)
    stopifnot(identical(history[[1L]]$requested_from, "reticulate"))
    stopifnot("numpy" %in% history[[1L]]$packages)
    stopifnot(identical(
      vapply(events, `[[`, character(1), "requested_from"),
      c("R_GlobalEnv", "stats", "graphics")
    ))
    stopifnot(identical(
      vapply(events, `[[`, character(1), "action"),
      c("add", "add", "remove")
    ))
    stopifnot(identical(
      lapply(events, `[[`, "packages"),
      list(
        "global-package",
        "diagnostic-package",
        "diagnostic-package"
      )
    ))
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
    warning <- tryCatch(
      package_py_require("not a valid requirement"),
      warning = conditionMessage
    )

    stopifnot(
      grepl(
        "Call `py_require()` to remove or replace conflicting requirements",
        warning,
        fixed = TRUE
      ),
      identical(before, py_require())
    )
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("freezing requirements pins an initialized managed Python", {
  session <- r_session(echo = FALSE, {
    library(reticulate)
    py_require(python_version = ">=3.10")

    testthat::with_mocked_bindings(
      py_write_requirements(
        packages = NULL,
        python_version = NULL,
        freeze = TRUE,
        python = NULL
      ),
      is_ephemeral_venv_initialized = function(python = NULL) TRUE,
      py_version = function(patch = FALSE) {
        stopifnot(patch)
        numeric_version("3.11.9")
      },
      uv_binary = function(...) "uv",
      resolve_python_version = function(constraints = NULL, uv = NULL) {
        stop("resolved: ", constraints, call. = FALSE)
      },
      .package = "reticulate"
    )
  })

  expect_match(session, "resolved: 3.11.9", fixed = TRUE, all = FALSE)
  expect_true(attr(session, "status", exact = TRUE) != 0L)
})

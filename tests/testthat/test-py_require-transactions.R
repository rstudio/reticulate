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
      c("python_version", "packages", "exclude_newer", "history")
    ))

    py_require(exclude_newer = "2020-01-01")
    py_require(exclude_newer = "", action = "remove")
    stopifnot(is.null(py_require()$exclude_newer))
    stopifnot(identical(
      names(py_require()),
      c("python_version", "packages", "exclude_newer", "history")
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
      c("python_version", "packages", "exclude_newer", "history")
    ))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printing names the default Python and source of requirements", {
  skip_if_not_installed("cli")

  session <- r_session(echo = FALSE, {
    library(reticulate)

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")
    package_py_require("example-package")

    printed <- capture.output(print(py_require()))
    stopifnot(startsWith(printed[[1L]], "═"))
    stopifnot(any(grepl("Current requirements", printed, fixed = TRUE)))
    stopifnot(any(grepl("Will default to", printed, fixed = TRUE)))
    stopifnot(any(grepl(
      "Python requirement requests (in order)",
      printed,
      fixed = TRUE
    )))
    stopifnot(any(grepl("R package stats", printed, fixed = TRUE)))
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("printing requirements has a base fallback", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    package_py_require <- function(...) reticulate::py_require(...)
    package_py_require <- rlang::zap_srcref(package_py_require)
    environment(package_py_require) <- asNamespace("stats")
    package_py_require("example-package")
    environment(package_py_require) <- asNamespace("graphics")
    package_py_require("example-package", action = "remove")

    require_namespace <- base::requireNamespace
    printed <- testthat::with_mocked_bindings(
      capture.output(print(py_require())),
      requireNamespace = function(package, ...) {
        if (identical(package, "cli"))
          FALSE
        else
          require_namespace(package, ...)
      },
      .package = "base"
    )
    stopifnot(grepl(
      "^=+ Python requirements =+$",
      printed[[1L]]
    ))
    stopifnot(any(grepl(
      "^-- Current requirements -+$",
      printed
    )))
    stopifnot(any(grepl(
      "^-- Python requirement requests \\(in order\\) -+$",
      printed
    )))
    stats <- grep("R package stats", printed, fixed = TRUE)
    graphics <- grep("R package graphics", printed, fixed = TRUE)
    add <- grep("Action: add", printed, fixed = TRUE)
    remove <- grep("Action: remove", printed, fixed = TRUE)
    stopifnot(length(stats) == 1L, length(graphics) == 1L)
    add <- add[add > stats][[1L]]
    remove <- remove[remove > graphics][[1L]]
    stopifnot(stats < add, add < graphics, graphics < remove)
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})

test_that("formatting requirements preserves its plain text layout", {
  session <- r_session(echo = FALSE, {
    library(reticulate)
    py_require(
      packages = c("numpy", "package-one", "package-two"),
      python_version = ">=3.10",
      action = "set"
    )

    formatted <- format(py_require(), width = 30L)
    stopifnot(identical(
      head(formatted, 5L),
      c(
        "Python requirements:",
        "  Python: >=3.10",
        "  Packages: numpy,",
        "            package-one,",
        "            package-two"
      )
    ))
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

    printed <- capture.output(print(requirements))
    stats <- grep("R package stats", printed, fixed = TRUE)
    graphics <- grep("R package graphics", printed, fixed = TRUE)
    add <- grep("Action: add", printed, fixed = TRUE)
    remove <- grep("Action: remove", printed, fixed = TRUE)
    stopifnot(length(stats) == 1L, length(graphics) == 1L)
    add <- add[add > stats][[1L]]
    remove <- remove[remove > graphics][[1L]]
    stopifnot(stats < add, add < graphics, graphics < remove)
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
    stopifnot(identical(before, after))
    stopifnot(identical(before_print, capture.output(print(after))))
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
        python = NULL,
        quiet = TRUE
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

test_that("py_require() accepts R date and date-time cutoffs", {
  session <- r_session(echo = FALSE, {
    library(reticulate)

    py_require(exclude_newer = as.Date("2006-12-02"))
    stopifnot(identical(py_require()$exclude_newer, "2006-12-02"))

    datetime <- as.POSIXct(
      "2006-12-02 02:07:43",
      tz = "America/New_York"
    ) + 0.125
    for (cutoff in list(datetime, as.POSIXlt(datetime))) {
      py_require(exclude_newer = cutoff, action = "set")
      stopifnot(identical(
        py_require()$exclude_newer,
        "2006-12-02T07:07:43Z"
      ))
    }
  })

  expect_null(attr(session, "status", exact = TRUE), info = paste(session, collapse = "\n"))
})


test_that("uv_run_tool() accepts R date and date-time cutoffs", {
  datetime <- as.POSIXct(
    "2006-12-02 02:07:43",
    tz = "America/New_York"
  ) + 0.125
  cutoffs <- list(
    as.Date("2006-12-02"),
    datetime,
    as.POSIXlt(datetime)
  )
  args <- list()
  testthat::with_mocked_bindings(
    for (cutoff in cutoffs) {
      uv_run_tool(
        "example",
        python_version = "exclude-newer-test",
        exclude_newer = cutoff
      )
    },
    uv_binary = function(...) "uv",
    resolve_python_version = function(...) "3.11",
    uv_exec = function(command, ...) {
      args[[length(args) + 1L]] <<- command
      0L
    },
    .package = "reticulate"
  )

  exclude_newer <- vapply(args, function(x) {
    x[[match("--exclude-newer", x) + 1L]]
  }, character(1))
  expect_identical(
    exclude_newer,
    c(
      "2006-12-02",
      rep("2006-12-02T07:07:43Z", 2L)
    )
  )
})


test_that("uv_get_or_create_env() accepts R date cutoffs", {
  args <- NULL
  testthat::with_mocked_bindings(
    reticulate:::uv_get_or_create_env(
      packages = NULL,
      python_version = "3.11",
      exclude_newer = as.Date("2006-12-02")
    ),
    uv_binary = function(...) "uv",
    resolve_python_version = function(...) "3.11",
    maybe_shQuote = identity,
    uv_exec = function(command, ...) {
      args <<- command
      writeLines("/tmp/python", command[[length(command)]])
      0L
    },
    .package = "reticulate"
  )

  expect_identical(
    args[[match("--exclude-newer", args) + 1L]],
    "2006-12-02"
  )
})

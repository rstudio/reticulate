transform_scrub_python_patch <- function(x) {
  x <- sub("3\\.([0-9]{1,2})\\.[0-9]{1,2}", "3.\\1.xx", x)
  # uv may report install timing depending on cache state; keep snapshots stable.
  x <- x[!grepl("^\\s*Installed [0-9]+ package(s)? in [0-9]+ms\\s*$", x)]
  x
}

expect_snapshot2 <- function(..., transform = transform_scrub_python_patch) {
  expect_snapshot(..., transform = transform)
}


test_that("Error requesting conflicting package versions", {
  local_edition(3)

  # dry run before snapshot tests, so download progress updates aren't
  # in the snapshot
  try(uv_get_or_create_env())

  expect_snapshot2(r_session({
    library(reticulate)
    py_require("numpy<2")
    py_require("numpy>=2")
    import("numpy")
    py_config()
  }))
})

test_that("Adding packages after Python init works; conflicting versions error", {
  local_edition(3)

  # dry run to pre-cache environments and avoid download messages in snapshot
  try(uv_get_or_create_env(c("numpy", "requests")))
  try(uv_get_or_create_env(c("numpy", "requests", "pandas")))

  expect_snapshot2(r_session({
    library(reticulate)
    py_require("numpy")
    import("sys") # force Python to initialize

    # happy path: adding a completely new package should work
    py_require("pandas")
    import("pandas")

    # adding a new package and the same version of an existing one should work
    py_require(c("numpy", "requests"))

    # error: adding a conflicting version of an already-required package
    try(py_require("numpy>2"))
  }))
})


test_that("Setting py_require(python_version) after initializing Python ", {
  test_py_require_reset()
  local_edition(3)
  # dry run to avoid installation messages in snapshot
  try(uv_get_or_create_env(c("numpy", "pandas"), "3.11"))
  try(uv_get_or_create_env(c("numpy", "pandas", "requests"), "3.11"))

  expect_snapshot2(r_session({
    Sys.unsetenv("RETICULATE_PYTHON")
    Sys.setenv("RETICULATE_USE_MANAGED_VENV" = "yes")
    # Sys.setenv("RETICULATE_PYTHON"="managed")

    pkg_py_require <- function(...) reticulate::py_require(...)
    pkg_py_require <- rlang::zap_srcref(pkg_py_require)
    environment(pkg_py_require) <- asNamespace("stats")

    library(reticulate)

    # multiple requests are fine
    py_require(python_version = ">=3.9", "pandas")
    py_require(python_version = ">=3.8,<3.14")
    py_require(python_version = "3.11")
    pkg_py_require(packages = c("pandas", "numpy"), python_version = ">=3.10")

    # initialize python, ensure packages are there
    prefix <- import("sys")$prefix
    import("numpy")
    import("pandas")
    stopifnot(py_version() == "3.11")

    # already satisfied requests are no-ops
    py_require(python_version = ">=3.9.1")
    py_require(python_version = ">=3.8.1,<3.14")
    py_require(python_version = "3.11")
    pkg_py_require(python_version = ">=3.10")
    py_require("numpy")
    py_require("pandas")
    py_require(c("numpy", "pandas"), action = "set")
    py_require(c("notexist"), action = "remove")

    try(import("requests"))

    # additional packages requests result in a new venv being activated
    py_require("requests")
    import("requests")
    prefix2 <- import("sys")$prefix
    stopifnot(prefix != prefix2)

    # unsatisfied requests from a package generate a warning
    # (it might make sense to narrow this to target just .onLoad() calls)
    pkg_py_require(python_version = ">=3.12")
    pkg_py_require("pandas", action = "remove")

    # unsatisfied requests called from outisde a package error
    try(py_require(exclude_newer = "2020-01-01"))
    try(py_require(python_version = ">=3.12"))
    try(py_require("pandas", action = "remove"))
  }))
})


test_that("Error requesting newer package version against an older snapshot", {
  session <- r_session(attach_namespace = TRUE, {
    uv_get_or_create_env(
      packages = "tensorflow==2.18.*",
      exclude_newer = "2024-10-20"
    )
  })
  expect_match(
    session,
    "Call `py_require()` to remove or replace conflicting requirements",
    fixed = TRUE,
    all = FALSE
  )
  expect_true(attr(session, "status", TRUE) != 0L)
})

test_that("Error requesting a package that does not exists", {
  local_edition(3)
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    py_require(c("pandas", "numpy", "notexists"))
    uv_get_or_create_env()
  }))
})

test_that("Error requesting conflicting Python versions", {
  local_edition(3)
  expect_snapshot2(
    r_session(attach_namespace = TRUE, {
      py_require(python_version = ">=3.10")
      py_require(python_version = "<3.10")
      uv_get_or_create_env()
    }),
    transform = function(x) {
      sub(
        "^Available Python versions found: 3\\.1[1-9]\\..*",
        "Available Python versions found: 3.11.xx ....",
        x
      )
    }
  )
})

test_that("Simple tests", {
  local_edition(3)
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require()
  }))
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require("numpy==2", action = "remove")
    py_require()
  }))
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require("numpy==2", action = "remove")
    py_require(exclude_newer = "1990-01-01")
    py_require()
  }))
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require("numpy==2", action = "remove")
    py_require(exclude_newer = "1990-01-01")
    py_require(python_version = c("<=3.11", ">=3.10"))
    py_require()
  }))
})

test_that("can bootstrap install uv in reticulate cache", {
  # This test needs rethinking. It assumes that uv is not already installed on the users system,
  # and fails if it is.
  if (Sys.which("uv") != "" || file.exists("~/.local/bin/uv")) {
    skip("uv installed by user")
  }
  local_edition(3)
  test_py_require_reset()
  uv_exec <- ifelse(is_windows(), "uv.exe", "uv")
  target_path <- reticulate_cache_dir("uv", "bin", uv_exec)
  expect_equal(
    normalizePath(target_path),
    normalizePath(uv_binary())
  )
})

test_that("Multiple py_require() calls from package are shows in one row", {
  local_edition(3)
  expect_snapshot2(r_session(attach_namespace = TRUE, {
    gr_package <- function() {
      py_require(paste0("package", 1:20))
      py_require(paste0("package", 1:10), action = "remove")
      py_require(python_version = c("<=3.11", ">=3.10"))
    }
    environment(gr_package) <- asNamespace("graphics")
    gr_package()
    py_require()
  }))
})

test_that("py_require() standard library module", {
  local_edition(3)
  expect_snapshot2(r_session({
    library(reticulate)
    py_require("os")

    os <- import("os")
  }))
})

test_that("py_require() warns missing packages in a virtual env", {
  local_edition(3)
  venv <- tempfile("venv")
  virtualenv_create(envname = venv)
  expr = bquote({
    library(reticulate)
    use_virtualenv(.(venv), required = TRUE)
    py_require("polars")

    config <- py_config()
  })
  expect_snapshot2(
    do.call(r_session, list(force_managed_python = FALSE, exprs = expr)),
    transform = function(x) {
      x <- transform_scrub_python_patch(x)
      # scrub paths
      gsub("[A-Za-z]:[\\\\/][^\"' ]+|/[A-Za-z0-9._/\\\\-]+", "***", x)
    }
  )
})

test_that("package load hooks get startup messages, not warnings", {
  expect_true(is_package_loading(list(quote(runHook(".onLoad", foo)))))
  expect_true(is_package_loading(list(quote(runHook(.onAttach, foo)))))
  expect_false(is_package_loading(list(quote(runHook(".onUnload", foo)))))

  msg <- paste(
    "Some Python package requirements declared via `py_require()` are not",
    "installed in the selected Python environment: (/path/to/python)\n",
    "  numpy"
  )

  expect_warning(
    warning_or_startup_message(msg, call. = FALSE),
    "Some Python package requirements declared via `py_require()` are not",
    fixed = TRUE
  )

  runHook <- function(name, expr) expr()
  expect_message(
    expect_warning(
      runHook(".onLoad", function() warning_or_startup_message(msg, call. = FALSE)),
      NA
    ),
    "Some Python package requirements declared via `py_require()` are not",
    fixed = TRUE
  )
  expect_message(
    expect_warning(
      runHook(".onAttach", function() warning_or_startup_message(msg, call. = FALSE)),
      NA
    ),
    "Some Python package requirements declared via `py_require()` are not",
    fixed = TRUE
  )
})

test_that("Error requesting conflicting package versions", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("numpy<2")
    py_require("numpy>=2")
    uv_get_or_create_env()
  }))
})

test_that("Error requesting newer package version against an older snapshot", {
  session <- r_session(attach_namespace = TRUE, {
    uv_get_or_create_env(
      packages = "tensorflow==2.18.*",
      exclude_newer = "2024-10-20"
    )
  })
  expect_match(session,
    "Call `py_require()` to remove or replace conflicting requirements",
    fixed = TRUE, all = FALSE
  )
  expect_true(attr(session, "status", TRUE) != 0L)
})

test_that("Error requesting a package that does not exists", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require(c("pandas", "numpy", "notexists"))
    uv_get_or_create_env()
  }))
})

test_that("Error requesting conflicting Python versions", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require(python_version = ">=3.10")
    py_require(python_version = "<3.10")
    uv_get_or_create_env()
  }))
})

test_that("Simple tests", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require()
  }))
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require("numpy==2", action = "remove")
    py_require()
  }))
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("pandas")
    py_require("numpy==2")
    py_require("numpy==2", action = "remove")
    py_require(exclude_newer = "1990-01-01")
    py_require()
  }))
  expect_snapshot(r_session(attach_namespace = TRUE, {
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
  if (Sys.which("uv") != "" || file.exists("~/.local/bin/uv"))
    skip("uv installed by user")
  local_edition(3)
  test_py_require_reset()
  uv_exec <- ifelse(is_windows(), "uv.exe", "uv")
  target_path <- path.expand(
    file.path(rappdirs::user_cache_dir("r-reticulate", NULL), "bin", uv_exec)
  )
  expect_equal(
    normalizePath(target_path),
    normalizePath(uv_binary())
  )
})

test_that("Multiple py_require() calls from package are shows in one row", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
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

test_that("'Equal to' and 'Non-equal to' Python requirements fail",{
  if (py_available()) {
    skip("Can't test py_require(python_version) declarations after python initialized")
    # TODO: fix test to actually test this
  }

  test_py_require_reset()
  py_require(python_version = "==3.11")
  x <- py_require()
  expect_equal(x$python_version, "3.11")
  expect_error(
    py_require(python_version = ">=3.10"),
    regexp = "Python version requirements cannot combine"
    )
  expect_error(
    py_require(python_version = "3.10"),
    regexp = "Python version requirements cannot contain"
  )
})



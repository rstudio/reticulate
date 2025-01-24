test_that("Error requesting conflicting package versions", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("numpy<2")
    py_require("numpy>=2")
    uv_get_or_create_env()
  }))
})

test_that("Error requesting newer package version against an older snapshot", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("tensorflow==2.18.*")
    py_require(exclude_newer = "2024-10-20")
    uv_get_or_create_env()
  }))
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
    py_require(python_version = c("3.11", ">=3.10"))
    py_require()
  }))
})

test_that("uv cache testing", {
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
      py_require(python_version = c("3.11", ">=3.10"))
    }
    environment(gr_package) <- asNamespace("graphics")
    gr_package()
    py_require()
  }))
})

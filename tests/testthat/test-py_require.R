test_that("Error requesting conflicting package versions", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("numpy<2")
    py_require("numpy>=2")
    get_or_create_venv()
  }))
})

test_that("Error requesting newer package version against an older snapshot", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require("tensorflow==2.18.*")
    py_require(exclude_newer = "2024-10-20")
    get_or_create_venv()
  }))
})

test_that("Error requesting a package that does not exists", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require(c("pandas", "numpy", "notexists"))
    get_or_create_venv()
  }))
})

test_that("Error requesting conflicting Python versions", {
  local_edition(3)
  expect_snapshot(r_session(attach_namespace = TRUE, {
    py_require(python_version = ">=3.10")
    py_require(python_version = "<3.10")
    get_or_create_venv()
  }))
})




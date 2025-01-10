



test_that("Error requesting newer package version against an older snapshot", {
  local_edition(3)
  withr::local_envvar(c("NO_COLOR" = 1))
  expect_snapshot(eval_remote(error = TRUE, {
    reticulate:::get_or_create_venv(c("numpy<2", "numpy>=2"))
  }))
})
  # test_py_require_reset()
  # py_require("tensorflow==2.18.*", exclude_newer = "2024-10-20")
  # expect_snapshot({
  #   get_or_create_venv("tensorflow==2.18.*", exclude_newer = "2024-10-20")
  #   get_or_create_venv("tensorflow==2.18.*")#, exclude_newer = "2024-10-20")
  # }, error = TRUE)
#
# test_that("Error requesting conflicting package versions", {
#   local_edition(3)
#   test_py_require_reset()
#   py_require("pandas==2.2.3")
#   py_require("pandas==2.2.2")
#   expect_snapshot(get_or_create_venv(), error = TRUE)
# })
#
# test_that("Error requesting conflicting Python versions", {
#   local_edition(3)
#   test_py_require_reset()
#   py_require(python_version = ">=3.10")
#   py_require(python_version = "3.11")
#   expect_snapshot(get_or_create_venv(), error = TRUE)
# })

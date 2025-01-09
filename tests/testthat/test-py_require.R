test_that("Error requesting newer package version against an older snapshot", {
  local_edition(3)
  py_require("tensorflow==2.18.*", exclude_newer = "2024-10-20")
  expect_snapshot(get_or_create_venv(), error = TRUE)
})

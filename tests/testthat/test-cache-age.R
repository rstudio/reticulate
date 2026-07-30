test_that("RETICULATE_MAX_CACHE_AGE configures automatic cache clearing", {
  skip_if(getRversion() <= "4.0")

  cache_root <- withr::local_tempdir()
  withr::local_envvar(c(
    R_USER_CACHE_DIR = cache_root,
    RETICULATE_MAX_CACHE_AGE = "0",
    RETICULATE_PYTHON = file.path(cache_root, "missing-python"),
    UV_OFFLINE = NA
  ))
  withr::local_options(reticulate.uv_binary = NULL)

  cache_dir <- tools::R_user_dir("reticulate", "cache")
  uv <- file.path(
    cache_dir,
    "uv",
    "bin",
    if (.Platform$OS.type == "windows") "uv.exe" else "uv"
  )
  dir.create(dirname(uv), recursive = TRUE)
  marker <- file.path(cache_dir, "uv", "marker")
  stopifnot(file.create(uv), file.create(marker))

  expect_message(
    virtualenv_starter(all = TRUE),
    "Clearing reticulate's uv cache",
    fixed = TRUE,
    all = FALSE
  )
  expect_false(file.exists(marker))
})

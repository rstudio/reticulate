test_that("expired uv cache is retained when the installer is unavailable", {
  skip_if(getRversion() <= "4.0")

  system_uv <- Sys.which("uv")
  skip_if(!nzchar(system_uv), "uv is not installed")
  version <- suppressWarnings(numeric_version(
    sub("uv ([0-9.]+).*", "\\1", system2(system_uv, "--version", stdout = TRUE)),
    strict = FALSE
  ))
  skip_if(anyNA(version) || version < "0.6.3", "uv is too old")

  cache_root <- withr::local_tempdir()
  withr::local_envvar(c(
    R_USER_CACHE_DIR = cache_root,
    RETICULATE_MAX_CACHE_AGE_DAYS = "-1",
    RETICULATE_UV = "managed",
    UV_OFFLINE = NA,
    UV_PYTHON_PREFERENCE = "only-managed"
  ))
  withr::local_options(
    download.file.method = "invalid",
    reticulate.max_cache_age = as.difftime(30, units = "days"),
    reticulate.uv_binary = NULL
  )

  cache_dir <- tools::R_user_dir("reticulate", "cache")
  uv <- file.path(
    cache_dir,
    "uv",
    "bin",
    if (.Platform$OS.type == "windows") "uv.exe" else "uv"
  )
  marker <- file.path(cache_dir, "uv", "marker")
  dir.create(dirname(uv), recursive = TRUE)
  stopifnot(file.copy(system_uv, uv), file.create(marker))

  expect_error(
    expect_message(
      uv_run_tool("example", python_version = ">999"),
      "Retaining reticulate's uv cache",
      fixed = TRUE,
      all = FALSE
    ),
    "could not be satisfied",
    fixed = TRUE
  )
  expect_true(file.exists(uv))
  expect_true(file.exists(marker))
})

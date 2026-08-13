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

test_that("uv_run_tool() rejects an unusable managed uv install", {
  skip_if(getRversion() <= "4.0")

  cache_root <- withr::local_tempdir()
  installer <- tempfile("install-uv-")
  withr::local_envvar(c(
    R_USER_CACHE_DIR = cache_root,
    RETICULATE_UV = "managed"
  ))
  withr::local_options(reticulate.uv_binary = NULL)

  testthat::with_mocked_bindings(
    expect_error(
      uv_run_tool("example"),
      "installed uv binary is not usable",
      fixed = TRUE
    ),
    download_uv_installer = function() installer,
    uv_install_managed = function(uv, installer) {
      rscript <- file.path(
        R.home("bin"),
        if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript"
      )
      dir.create(dirname(uv), recursive = TRUE)
      stopifnot(file.copy(rscript, uv))
    },
    resolve_python_version = function(...) {
      stop("unusable uv reached the resolver", call. = FALSE)
    },
    .package = "reticulate"
  )

})

test_that("uv_run_tool() caches a managed uv without rebuilding its path", {
  for (configured_from_env in c(TRUE, FALSE)) local({
    cache_root <- withr::local_tempdir()
    withr::local_envvar(
      RETICULATE_UV = if (configured_from_env) "managed" else NA
    )
    withr::local_options(
      reticulate.uv_binary = if (configured_from_env) NULL else "managed"
    )

    path_calls <- usable_calls <- 0L
    testthat::with_mocked_bindings(
      {
        expect_error(
          uv_run_tool("example", python_version = "cache-fast-path"),
          "resolved",
          fixed = TRUE
        )
        expect_error(
          uv_run_tool("example", python_version = "cache-fast-path"),
          "resolved",
          fixed = TRUE
        )
      },
      reticulate_cache_dir = function(...) {
        path_calls <<- path_calls + 1L
        file.path(cache_root, ...)
      },
      maybe_clear_reticulate_uv_cache = function(...) NULL,
      uv_is_usable = function(...) {
        usable_calls <<- usable_calls + 1L
        TRUE
      },
      resolve_python_version = function(constraints = NULL, uv = NULL) {
        stopifnot(isTRUE(attr(uv, "reticulate-managed", exact = TRUE)))
        stop("resolved", call. = FALSE)
      },
      .package = "reticulate"
    )

    expect_identical(path_calls, 1L)
    expect_identical(usable_calls, 1L)
    if (configured_from_env)
      expect_identical(Sys.getenv("RETICULATE_UV"), "managed")
  })
})

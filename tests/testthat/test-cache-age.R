local_expired_uv_cache <- function(max_cache_age = NULL,
                                   reticulate_uv = NA_character_) {
  local_envir <- parent.frame()
  cache_root <- withr::local_tempdir(.local_envir = local_envir)
  home <- withr::local_tempdir(.local_envir = local_envir)
  bin_dir <- withr::local_tempdir(.local_envir = local_envir)

  withr::local_envvar(c(
    R_USER_CACHE_DIR = cache_root,
    RETICULATE_MAX_CACHE_AGE_DAYS = "0",
    RETICULATE_PYTHON = file.path(cache_root, "missing-python"),
    RETICULATE_UV = reticulate_uv,
    UV_OFFLINE = NA,
    UV_PYTHON_PREFERENCE = "only-managed",
    HOME = home,
    USERPROFILE = home,
    PATH = bin_dir
  ), .local_envir = local_envir)

  options <- list(reticulate.uv_binary = NULL)
  if (!is.null(max_cache_age))
    options$reticulate.max_cache_age <- max_cache_age
  withr::local_options(options, .local_envir = local_envir)

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

  state <- new.env(parent = emptyenv())
  state$bin_dir <- bin_dir
  state$downloads <- 0L
  state$installer <- NULL
  state$marker <- marker
  state$uv <- uv
  state
}

mock_installer_download <- function(state, succeeds = TRUE) {
  force(state)
  force(succeeds)
  function(url, destfile, ...) {
    state$downloads <- state$downloads + 1L
    state$installer <- destfile
    writeLines("installer", destfile)
    if (!succeeds)
      stop("offline", call. = FALSE)
    0L
  }
}

same_file_path <- function(x, y) {
  identical(
    normalizePath(x, winslash = "/", mustWork = FALSE),
    normalizePath(y, winslash = "/", mustWork = FALSE)
  )
}

mock_uv_system2 <- function(state, python_version, install = FALSE) {
  force(state)
  force(python_version)
  force(install)
  version_parts <- strsplit(python_version, ".", fixed = TRUE)[[1L]]
  stopifnot(length(version_parts) == 3L)
  python_list <- paste0(
    '[{"version":"', python_version, '",',
    '"version_parts":{"major":', version_parts[[1L]],
    ',"minor":', version_parts[[2L]],
    ',"patch":', version_parts[[3L]], '},',
    '"path":null,"symlink":null,',
    '"url":"https://example.invalid/python",',
    '"variant":"default","implementation":"cpython"}]'
  )
  function(command, args = character(), ...) {
    command <- as.character(command)
    if (same_file_path(command, state$uv)) {
      if (identical(args, "--version"))
        return("uv 999.0.0")
      if (any(grepl("python list", args, fixed = TRUE)))
        return(python_list)
      return(0L)
    }

    is_installer <- !is.null(state$installer) &&
      if (.Platform$OS.type == "windows") {
        identical(tolower(basename(command)), "powershell.exe") &&
          utils::shortPathName(state$installer) %in% args
      } else {
        same_file_path(command, state$installer)
      }
    if (is_installer && install) {
      dir.create(dirname(state$uv), recursive = TRUE, showWarnings = FALSE)
      stopifnot(file.create(state$uv))
      return(0L)
    }

    stop("unexpected system2 call: ", command, call. = FALSE)
  }
}

test_that("RETICULATE_MAX_CACHE_AGE_DAYS takes precedence over the R option", {
  skip_if(getRversion() <= "4.0")

  state <- local_expired_uv_cache(as.difftime(30, units = "days"))
  result <- testthat::with_mocked_bindings(
    testthat::with_mocked_bindings(
      expect_message(
        uv_run_tool("example", python_version = "3.12.999"),
        "Clearing reticulate's uv cache",
        fixed = TRUE,
        all = FALSE
      ),
      system2 = mock_uv_system2(state, "3.12.999", install = TRUE),
      .package = "base"
    ),
    download.file = mock_installer_download(state),
    .package = "reticulate"
  )

  expect_equal(result, 0L)
  expect_equal(state$downloads, 1L)
  expect_false(file.exists(state$installer))
  expect_false(file.exists(state$marker))
})

test_that("expired uv cache is retained when the uv installer is unavailable", {
  skip_if(getRversion() <= "4.0")

  state <- local_expired_uv_cache()
  result <- testthat::with_mocked_bindings(
    testthat::with_mocked_bindings(
      expect_message(
        uv_run_tool("example", python_version = "3.12.998"),
        "Retaining reticulate's uv cache",
        fixed = TRUE,
        all = FALSE
      ),
      system2 = mock_uv_system2(state, "3.12.998"),
      .package = "base"
    ),
    download.file = mock_installer_download(state, succeeds = FALSE),
    .package = "reticulate"
  )

  expect_equal(result, 0L)
  expect_equal(state$downloads, 1L)
  expect_false(file.exists(state$installer))
  expect_true(file.exists(state$uv))
  expect_true(file.exists(state$marker))
})

test_that("uv availability probes do not clear the expired cache", {
  skip_if(getRversion() <= "4.0")

  for (reticulate_uv in list(NA_character_, "managed")) {
    local({
      python_version <- if (is.na(reticulate_uv)) "3.12.997" else "3.12.996"
      state <- local_expired_uv_cache(reticulate_uv = reticulate_uv)
      system2 <- base::system2
      uv_version_checks <- 0L

      result <- testthat::with_mocked_bindings(
        {
          testthat::with_mocked_bindings(
            virtualenv_starter(all = TRUE),
            system2 = function(command, args = character(), ...) {
              if (same_file_path(as.character(command), state$uv) &&
                  identical(args, "--version")) {
                uv_version_checks <<- uv_version_checks + 1L
                return("uv 999.0.0")
              }
              system2(command, args, ...)
            },
            .package = "base"
          )

          expect_equal(state$downloads, 0L)
          expect_equal(uv_version_checks, 1L)
          expect_true(file.exists(state$marker))

          testthat::with_mocked_bindings(
            uv_run_tool("example", python_version = python_version),
            system2 = mock_uv_system2(
              state,
              python_version,
              install = TRUE
            ),
            .package = "base"
          )
        },
        download.file = mock_installer_download(state),
        .package = "reticulate"
      )

      expect_equal(result, 0L)
      expect_equal(state$downloads, 1L)
      expect_false(file.exists(state$installer))
      expect_false(file.exists(state$marker))
    })
  }
})

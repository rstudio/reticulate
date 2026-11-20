# Run in a fresh Rscript session with the x64 and ARM64 Python executables.
stopifnot(.Platform$OS.type == "windows", R.version$arch == "aarch64")
pythons <- commandArgs(trailingOnly = TRUE)
stopifnot(length(pythons) == 2L, all(file.exists(pythons)))

Sys.setenv(RETICULATE_USE_MANAGED_VENV = "no")

# With only x64 Python on PATH, discover ARM64 Python from the registry.
config <- withr::with_path(dirname(pythons[[1L]]), action = "replace",
                          reticulate::py_discover_config())
registered <- subset(reticulate::py_versions_windows(), type == "PythonCore" & arch == "arm64")
stopifnot(identical(config$forced, "Windows Registry"), identical(config$platform, "win-arm64"),
          any(registered$version == as.character(config$version) &
              normalizePath(registered$executable_path) == normalizePath(config$python)))

# Give the two PATH candidates independent installations of NumPy.
venvs <- file.path(tempdir(), c("x64", "arm64"))
for (i in seq_along(pythons))
  stopifnot(system2(pythons[[i]], c("-m", "venv", shQuote(venvs[[i]]))) == 0L)

bins <- file.path(venvs, "Scripts")
python3 <- file.path(bins[[1L]], "python3.exe")
python <- file.path(bins[[2L]], "python.exe")
stopifnot(file.rename(file.path(bins[[1L]], "python.exe"), python3))
withr::local_path(bins)
stopifnot(file.exists(python), file.exists(python3),
          normalizePath(Sys.which("python3")) == normalizePath(python3),
          normalizePath(Sys.which("python")) == normalizePath(python))

discover <- function() normalizePath(reticulate::py_discover_config()$python)
without_numpy <- discover()
for (exe in c(python3, python))
  stopifnot(system2(exe, c("-m", "pip", "install", "--only-binary=:all:", "numpy")) == 0L)
with_numpy <- discover()

stopifnot(with_numpy == normalizePath(python), without_numpy == normalizePath(python))

Sys.setenv(RETICULATE_PYTHON = python3)
error <- tryCatch(reticulate::py_config(), error = conditionMessage)
stopifnot(is.character(error), grepl("win-arm64", error), grepl("win-amd64", error))

Sys.setenv(RETICULATE_PYTHON = python)
stopifnot(reticulate::py_eval("__import__('sysconfig').get_platform()") == "win-arm64")

#!/usr/bin/env Rscript

package_dir <- Sys.getenv("R_PACKAGE_DIR", NA)
if (is.na(package_dir))
  package_dir <- path.package("reticulate")

tryCatch({
  owd <- setwd(package_dir)
  for (stale_pycache in grep("__pycache__$", list.dirs(recursive = TRUE), value = TRUE))
    unlink(stale_pycache, recursive = TRUE)

  df <- reticulate::virtualenv_starter(all = TRUE)
  df <- df[order(df$version, decreasing = TRUE), ]
  df$minor <- df$version[, 1:2]
  df <- df[!duplicated(df$minor), ]
  for (python in df$path) {
    reticulate:::system2t(python, "-m compileall config python")
  }
}, finally = setwd(owd))

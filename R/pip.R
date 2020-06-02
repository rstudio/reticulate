
pip_version <- function(python) {

  # if we don't have pip, just return a placeholder version
  if (!file.exists(python))
    return(numeric_version("0.0"))

  # otherwise, ask pip what version it is
  command <- "import sys; import pip; sys.stdout.write(pip.__version__)"
  version <- system2(python, c("-c", shQuote(command)), stdout = TRUE, stderr = TRUE)
  
  # remove any beta components from version string
  idx <- regexpr("[[:alpha:]]", version)
  if (idx != -1)
    version <- substring(version, 1, idx - 1)

  # construct numeric version  
  numeric_version(version)

}

pip_install <- function(python, packages, pip_options = character(), ignore_installed = FALSE) {

  # construct command line arguments
  args <- c("-m", "pip", "install", "--upgrade")
  if (ignore_installed)
    args <- c(args, "--ignore-installed")
  args <- c(args, pip_options)
  args <- c(args, packages)

  # run it
  result <- system2(python, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error installing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }

  invisible(packages)

}

pip_uninstall <- function(python, packages) {

  # run it
  args <- c("-m", "pip", "uninstall", "--yes", packages)
  result <- system2(python, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error removing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }

  packages

}

pip_freeze <- function(python) {
  
  args <- c("-m", "pip", "freeze")
  output <- system2(python, args, stdout = TRUE)
  splat <- strsplit(output, "==| @ ")
  packages <- vapply(splat, `[[`, 1L, FUN.VALUE = character(1))
  versions <- vapply(splat, `[[`, 2L, FUN.VALUE = character(1))
  data.frame(
    package     = packages,
    version     = versions,
    requirement = paste(packages, versions, sep = "=="),
    stringsAsFactors = FALSE
  )
  
}

pip_version <- function(pip) {

  # if we don't have pip, just return a placeholder version
  if (!file.exists(pip))
    return(numeric_version("0.0"))

  # otherwise, ask pip what version it is
  output <- system2(pip, "--version", stdout = TRUE)
  parts <- strsplit(output, "\\s+")[[1]]
  version <- parts[[2]]
  numeric_version(version)

}


pip_install <- function(pip, packages, ignore_installed = TRUE) {

  # construct command line arguments
  args <- c("install", "--upgrade")
  if (ignore_installed)
    args <- c(args, "--ignore-installed")
  args <- c(args, packages)

  # run it
  result <- system2(pip, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error installing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }

  invisible(packages)

}

pip_uninstall <- function(pip, packages) {

  # run it
  args <- c("uninstall", "--yes", packages)
  result <- system2(pip, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error removing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }

  packages

}

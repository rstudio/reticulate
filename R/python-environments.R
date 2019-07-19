
python_environment_resolve <- function(envname = NULL, resolve = identity) {

  # use RETICULATE_PYTHON_ENV as default
  envname <- envname %||% Sys.getenv("RETICULATE_PYTHON_ENV", unset = "r-reticulate")

  # treat environment 'names' containing slashes as full paths
  if (grepl("[/\\]", envname)) {
    envname <- normalizePath(envname, winslash = "/", mustWork = FALSE)
    return(envname)
  }

  # otherwise, resolve the environment name as necessary
  resolve(envname)

}

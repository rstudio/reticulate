error_no_toml_parser = function(e) {
  stop("Please install 'RcppTOML' to parse the a poetry pyproject.toml", call. = FALSE)
}

pyproject_parser = function() {
  # From: https://github.com/rstudio/blogdown/blob/main/R/utils.R#L343
  tryCatch(getFromNamespace('parseTOML', 'RcppTOML'), error = error_no_toml_parser)
}
  
pyproject_path <- function() {
  # check option
  pyproject <- getOption("reticulate.poetry.pyproject")
  if (!is.null(pyproject))
    return(pyproject)
  
  # try default
  tryCatch(here::here("pyproject.toml"), error = function(e) "")
}

is_poetry_project <- function() {
  path = pyproject_path()
  
  # If there is no pyproject.toml, it is not a poetry project
  if (!file.exists(path)) return(FALSE)
  
  # read the pyproject.toml
  parser = pyproject_parser()
  toml = parser(path, fromFile = TRUE)
  
  # If poetry is listed as a pyproject, it is a poetry project
  "poetry" %in% names(toml[["tool"]])
}

poetry_python <- function() {
  
  # validate that poetry is available on the PATH
  if (!nzchar(Sys.which("poetry")))
    stop("'poetry' is not available")
  
  # move to root directory
  root <- here::here()
  owd <- setwd(root)
  on.exit(setwd(owd), add = TRUE)
  
  # ask poetry what the environment path is
  envpath <- system("poetry env info --path", intern = TRUE)
  status <- attr(envpath, "status") %||% 0L
  if (status != 0L) {
    fmt <- "'poetry env info --path' had status %i"
    stopf(fmt, status)
  }
  
  # get path to python
  virtualenv_python(envpath)
}

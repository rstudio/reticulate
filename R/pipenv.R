
pipenv_pipfile_path <- function() {
  
  # check option
  pipfile <- getOption("reticulate.pipenv.pipfile")
  if (!is.null(pipfile))
    return(pipfile)
  
  # try default
  tryCatch(
    here::here("Pipfile"),
    error = function(e) ""
  )
  
}

pipenv_python <- function() {
  
  # validate that pipenv is available on the PATH
  if (!nzchar(Sys.which("pipenv")))
    stop("'pipenv' is not available")
  
  # move to root directory
  root <- here::here()
  owd <- setwd(root)
  on.exit(setwd(owd), add = TRUE)
  
  # ask pipenv what the environment path is
  envpath <- system("pipenv --venv", intern = TRUE)
  status <- attr(envpath, "status") %||% 0L
  if (status != 0L) {
    fmt <- "'pipenv --venv' had status %i"
    stopf(fmt, status)
  }
  
  # get path to python
  virtualenv_python(envpath)
  
}

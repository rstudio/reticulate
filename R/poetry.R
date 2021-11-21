
poetry_binary_path <- function() {
  
  poetry <- getOption("reticulate.poetry.path")
  if (!is.null(poetry))
    return(poetry)
  
  Sys.which("poetry")
  
}

poetry_project_path <- function() {
  
  # check option
  project <- getOption("reticulate.poetry.project")
  if (!is.null(project))
    return(project)
    
  # try default
  tryCatch(
    here::here("pyproject.toml"),
    error = function(e) ""
  )
    
}

poetry_python_path <- function(project) {
  
  # move to project directory
  owd <- setwd(project)
  on.exit(setwd(owd), add = TRUE)
  
  # ask poetry where the virtual environment lives
  envpath <- system2("poetry", c("env", "info", "--path"), stdout = TRUE)
  
  # resolve python from this path
  virtualenv_python(envpath)

}

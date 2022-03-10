
poetry_config <- function(required_module) {

  # check for project file
  project <- poetry_project()
  projfile <- file.path(project, "pyproject.toml")
  if (!file.exists(projfile))
    return(NULL)

  # try to read it
  toml <- tryCatch(
    RcppTOML::parseTOML(projfile),
    error = identity
  )

  if (inherits(toml, "error")) {
    warning("This project contains a 'pyproject.toml' file, but it could not be parsed")
    warning(toml)
    return(NULL)
  }

  # check that it has a 'tool.poetry' section
  info <- tryCatch(toml[[c("tool", "poetry")]], error = identity)
  if (inherits(info, "error"))
    return(NULL)

  # validate that 'poetry' is available
  poetry <- poetry_exe()
  if (!file.exists(poetry)) {

    msg <- heredoc("
      This project appears to use Poetry for Python dependency management.
      However, the 'poetry' command line tool is not available.
      reticulate will be unable to activate this project.
      Please ensure that 'poetry' is available on the PATH.
    ")

    warning(msg)
    return(NULL)

  }

  python <- poetry_python(project)
  python_config(python, required_module, forced = "Poetry")

}

poetry_exe <- function() {

  poetry <- getOption("reticulate.poetry.exe")
  if (!is.null(poetry))
    return(poetry)

  Sys.which("poetry")

}

poetry_project <- function() {

  # check option
  project <- getOption("reticulate.poetry.project")
  if (!is.null(project))
    return(project)

  # try default
  projfile <- tryCatch(
    dirname(here::here("pyproject.toml")),
    error = function(e) ""
  )

}

poetry_python <- function(project) {

  # move to project directory
  owd <- setwd(project)
  on.exit(setwd(owd), add = TRUE)

  # ask poetry where the virtual environment lives
  envpath <- system2("poetry", c("env", "info", "--path"), stdout = TRUE)

  # resolve python from this path
  virtualenv_python(envpath)

}

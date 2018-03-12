
#' Configure which version of Python to use
#'
#' @param python Path to Python binary
#' @param virtualenv Directory of Python virtualenv
#' @param condaenv Name of Conda environment
#' @param conda Conda executable. Default is `"auto"`, which checks the `PATH`
#'   as well as other standard locations for Anaconda installations.
#' @param required Is this version of Python required? If `TRUE` then
#'  an error occurs if it's not located. Otherwise, the version is taken
#'  as a hint only and scanning for other versions will still proceed.
#'
#' @importFrom utils file_test
#'
#' @export
use_python <- function(python, required = FALSE) {

  if (required && !file_test("-f", python) && !file_test("-d", python))
    stop("Specified version of python '", python, "' does not exist.")

  if (required)
    .globals$required_python_version <- python
  
  .globals$use_python_versions <- unique(c(.globals$use_python_versions, python))
}


#' @rdname use_python
#' @export
use_virtualenv <- function(virtualenv, required = FALSE) {

  # prepend root virtualenv directory it doesn't exist and 
  # it's not an absolute path
  if (!utils::file_test("-d", virtualenv) & 
      !grepl("^/|^[a-zA-Z]:/|^~", virtualenv, perl = TRUE)) {
    workon_home <- Sys.getenv("WORKON_HOME", unset = "~/.virtualenvs")
    virtualenv <- file.path(workon_home, virtualenv)
  }
  
  # compute the bin dir
  if (is_windows())
    python_dir <- file.path(virtualenv, "Scripts")
  else
    python_dir <- file.path(virtualenv, "bin")
  

  # validate it if required
  if (required) {
    if (!file_test("-d", python_dir) ||
        !file_test("-f", file.path(python_dir, "activate_this.py"))) {
      stop("Directory ", virtualenv, " is not a Python virtualenv")
    }
  }

  # set the option
  python <- file.path(python_dir, "python")
  if (is_windows())
    python <- paste0(python, ".exe")
  use_python(python, required = required)
}

#' @rdname use_python
#' @export
use_condaenv <- function(condaenv, conda = "auto", required = FALSE) {

  # list all conda environments
  conda_envs <- conda_list(conda)
  
  # look for one with that name
  conda_env_python <- subset(conda_envs, conda_envs$name == condaenv)$python
  if (length(conda_env_python) == 0 && required)
    stop("Unable to locate conda environment '", condaenv, "'.")
  
  if (!is.null(condaenv))
    use_python(conda_env_python, required = required)
  
  invisible(NULL)
}



#' Interface to conda utility commands
#' 
#' @param envname Name of conda environment
#' @param conda Path to conda executable (or "auto" to find conda using the PATH
#'   and other conventional install locations).
#' @param packages Character vector with package names to install.
#' @param pip `TRUE` to use pip (defaults to `FALSE`)
#'   
#' @return `conda_list()` returns a data frame with the names and paths to the
#'   respective python binaries of available environments. `conda_create()`
#'   returns the Path to the python binary of the created environment.
#'   `conda_binary()` returns the location of the main conda binary or `NULL`
#'   if none can be found.
#'   
#' @keywords intername
#' @name conda-tools
#' 
#' @importFrom jsonlite fromJSON
#'   
#' @export
conda_list <- function(conda = "auto") {
  
  # resolve conda binary
  conda <- conda_binary(conda)
  
  # list envs
  conda_envs <- suppressWarnings(
    system2(conda, args = c("info", "--json"), stdout = TRUE)
  )
  
  # check for error
  status <- attr(conda_envs, "status")
  if (!is.null(status)) {
    # show warning if conda_diagnostics are enabled
    if (getOption("reticulate.conda_diagnostics", default = FALSE)) {
      errmsg <- attr(status, "errmsg")
      warning("Error ", status, " occurred running ", conda, " ", errmsg)
    }
    # return empty data frame
    return(data.frame(
      name = character(), 
      python = character(), 
      stringsAsFactors = FALSE)
    )
  }
  
  # strip out anaconda cloud prefix (not valid json)
  if (length(conda_envs) > 0 && grepl("Anaconda Cloud", conda_envs[[1]], fixed = TRUE))
    conda_envs <- conda_envs[-1]
    
  # convert to json
  conda_envs <- fromJSON(conda_envs)$envs

  # build data frame
  name <- character()
  python <- character()
  for (conda_env in conda_envs) {
    name <- c(name, basename(conda_env))
    conda_env_dir <- conda_env
    if (!is_windows())
      conda_env_dir <- file.path(conda_env_dir, "bin")
    conda_env_python <- file.path(conda_env_dir, "python")
    if (is_windows()) {
      conda_env_python <- paste0(conda_env_python, ".exe")
      conda_env_python <- normalizePath(conda_env_python)
    }
    python <- c(python, conda_env_python)
    
  }
  data.frame(name = name, python = python, stringsAsFactors = FALSE)
}



#' @rdname conda-tools
#' @export
conda_create <- function(envname, packages = "python", conda = "auto") {

  # resolve conda binary
  conda <- conda_binary(conda)
  
  # create the environment 
  result <- system2(conda, shQuote(c("create", "--yes", "--name", envname, packages)))
  if (result != 0L) {
    stop("Error ", result, " occurred creating conda environment ", envname,
         call. = FALSE)
  }
  
  # return the path to the python binary
  conda_envs <- conda_list(conda)
  invisible(subset(conda_envs, conda_envs$name == envname)$python)
}

#' @rdname conda-tools
#' @export
conda_remove <- function(envname, packages = NULL, conda = "auto") {
  
  # resolve conda binary
  conda <- conda_binary(conda)
  
  # no packages means everything
  if (is.null(packages))
    packages <- "--all"
  
  # remove packges (or the entire environment)
  result <- system2(conda, shQuote(c("remove", "--yes", "--name", envname, packages)))
  if (result != 0L) {
    stop("Error ", result, " occurred removing conda environment ", envname,
         call. = FALSE)
  }
}

#' @param pip_ignore_installed Ignore installed versions when using pip. This is `TRUE` by default
#'   so that specific package versions can be installed even if they are downgrades. The `FALSE` 
#'   option is useful for situations where you don't want a pip install to attempt an overwrite
#'   of a conda binary package (e.g. SciPy on Windows which is very difficult to install via
#'   pip due to compilation requirements).
#'
#' @rdname conda-tools
#' 
#' @keywords internal
#' 
#' @export
conda_install <- function(envname, packages, pip = FALSE, pip_ignore_installed = TRUE, conda = "auto") {
 
  # resolve conda binary
  conda <- conda_binary(conda)
  
  if (pip) {
    # use pip package manager
    condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
    cmd <- sprintf("%s%s %s && pip install --upgrade %s %s%s",
                   ifelse(is_windows(), "", ifelse(is_osx(), "source ", "/bin/bash -c \"source ")),
                   shQuote(path.expand(condaenv_bin("activate"))),
                   envname,
                   ifelse(pip_ignore_installed, "--ignore-installed", ""),
                   paste(shQuote(packages), collapse = " "),
                   ifelse(is_windows(), "", ifelse(is_osx(), "", "\"")))
    result <- system(cmd)
    
  } else {
    # use native conda package manager
    result <- system2(conda, shQuote(c("install", "--yes", "--name", envname, packages)))
  }
  
  # check for errors
  if (result != 0L) {
    stop("Error ", result, " occurred installing packages into conda environment ", 
         envname, call. = FALSE)
  }
  
  invisible(NULL)
}


#' @rdname conda-tools
#' @export
conda_binary <- function(conda = "auto") {
  
  # automatic lookup if requested
  if (identical(conda, "auto")) {
    conda <- find_conda()
    if (is.null(conda))
      stop("Unable to find conda binary. Is Anaconda installed?", call. = FALSE)
    conda <- conda[[1]]
  }
  
  # validate existence
  if (!file.exists(conda))
    stop("Specified conda binary '", conda, "' does not exist.", call. = FALSE)
  
  # return conda
  conda
}


#' @rdname conda-tools
#' @export
conda_version <- function(conda = "auto") {
  conda_bin <- conda_binary(conda)
  system2(conda_bin, "--version", stdout = TRUE)
}



find_conda <- function() {
  conda <- Sys.which("conda")
  if (!nzchar(conda)) {
    conda_locations <- c(
      path.expand("~/anaconda/bin/conda"),
      path.expand("~/anaconda2/bin/conda"),
      path.expand("~/anaconda3/bin/conda"),
      path.expand("~/anaconda4/bin/conda"),
      path.expand("~/miniconda/bin/conda"),
      path.expand("~/miniconda2/bin/conda"),
      path.expand("~/miniconda3/bin/conda"),
      path.expand("~/miniconda4/bin/conda"),
      path.expand("/anaconda/bin/conda"),
      path.expand("/anaconda2/bin/conda"),
      path.expand("/anaconda3/bin/conda"),
      path.expand("/anaconda4/bin/conda"),
      path.expand("/miniconda/bin/conda"),
      path.expand("/miniconda2/bin/conda"),
      path.expand("/miniconda3/bin/conda"),
      path.expand("/miniconda4/bin/conda")
    )
    if (is_windows()) {
      anaconda_versions <- windows_registry_anaconda_versions()
      anaconda_versions <- subset(anaconda_versions, anaconda_versions$arch == .Platform$r_arch)
      if (nrow(anaconda_versions) > 0) {
        conda_scripts <- utils::shortPathName(
          file.path(anaconda_versions$install_path, "Scripts", "conda.exe")
        )
        conda_locations <- c(conda_locations, conda_scripts)
      }
    }
    conda_locations <- conda_locations[file.exists(conda_locations)]
    if (length(conda_locations) > 0)
      conda_locations
    else
      NULL
  } else {
    conda
  }
}





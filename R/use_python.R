
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

  .globals$use_python_versions <- unique(c(.globals$use_python_versions, python))
}


#' @rdname use_python
#' @export
use_virtualenv <- function(virtualenv, required = FALSE) {

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
  use_python(python)
}

#' @rdname use_python
#' @export
use_condaenv <- function(condaenv, conda = "auto", required = FALSE) {

  # list all conda environments
  conda_envs <- conda_list(conda)
  
  # look for one with that name
  conda_env_python <- subset(conda_envs, conda_envs$name == condaenv)$python
  if (is.null(conda_env_python) && required)
    stop("Unable to locate conda environment '", condaenv, "'.")
  
  if (!is.null(condaenv))
    use_python(conda_env_python)
  
  invisible(NULL)
}



#' Interface to conda utility commands
#' 
#' @param envname Name of conda environment
#' @param conda Path to conda executable (or "auto" to find conda using the PATH
#'   and other conventional install locations).
#' @param pkgs Character vector with package names to install.
#' @param pip `TRUE` to use pip (defaults to `FALSE`)
#'   
#' @return `conda_list()` returns a data frame with the names and paths to the
#'   respective python binaries of available environments. `conda_create()`
#'   returns the Path to the python binary of the created environment.
#'   
#' @keywords intername
#' @name conda-tools
#'   
#' @export
conda_list <- function(conda = "auto") {
  
  # resolve conda binary
  conda <- resolve_conda(conda)
  
  # list envs
  conda_envs <- system2(conda, args = c("info", "--envs"), stdout = TRUE)
  matches <- regexec(paste0("^([^#][^ ]+)[ \\*]+(.*)$"), conda_envs)
  matches <- regmatches(conda_envs, matches)
  
  # build data frame
  name <- character()
  python <- character()
  for (match in matches) {
    if (length(match) == 3) {
      name <- c(name, match[[2]])
      conda_env_dir <- match[[3]]
      if (!is_windows())
        conda_env_dir <- file.path(conda_env_dir, "bin")
      conda_env_python <- file.path(conda_env_dir, "python")
      if (is_windows()) {
        conda_env_python <- paste0(conda_env_python, ".exe")
        conda_env_python <- normalizePath(conda_env_python)
      }
      python <- c(python, conda_env_python)
    }
  }
  data.frame(name = name, python = python, stringsAsFactors = FALSE)
}



#' @rdname conda-tools
#' @export
conda_create <- function(envname, conda = "auto") {

  # resolve conda binary
  conda <- resolve_conda(conda)
  
  # create the environment
  result <- system2(conda, shQuote(c("create", "--yes", "--name", envname)))
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
conda_install <- function(envname, pkgs, pip = FALSE, conda = "auto") {
 
  # resolve conda binary
  conda <- resolve_conda(conda)
  
  if (pip) {
    # use pip package manager
    condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
    cmd <- sprintf("%s%s && %s install --upgrade %s",
                   ifelse(is_windows(), "", "source "),
                   shQuote(path.expand(condaenv_bin("activate"))),
                   shQuote(path.expand(condaenv_bin("pip"))),
                   paste(shQuote(pkgs), collapse = " "))
    result <- system(cmd)
    
  } else {
    # use native conda package manager
    result <- system2(conda, shQuote(c("install", "--yes", "--name", envname, pkgs)))
  }
  
  # check for errors
  if (result != 0L) {
    stop("Error ", result, " occurred installing packages into conda environment ", 
         envname, call. = FALSE)
  }
  
  invisible(NULL)
}




resolve_conda <- function(conda) {
  
  # automatic lookup if requested
  if (identical(conda, "auto")) {
    conda = find_conda()
    if (is.null("conda"))
      stop("Unable to find conda binary. Is Anaconda installed?", call. = FALSE)
    conda <- conda[[1]]
  }
  
  # validate existence
  if (!file.exists(conda))
    stop("Specified conda binary '", conda, "' does not exist.", call. = FALSE)
  
  # return conda
  conda
}

find_conda <- function() {
  conda <- Sys.which("conda")
  if (!nzchar(conda)) {
    conda_locations <- c(
      path.expand("~/anaconda/bin/conda"),
      path.expand("~/anaconda3/bin/conda")
    )
    if (is_windows()) {
      anaconda_versions <- read_anaconda_versions_from_registry()
      if (length(anaconda_versions) > 0) {
        conda_scripts <- file.path(dirname(anaconda_versions), "Scripts", "conda.exe")
        conda_locations <- c(conda_locations, conda_scripts)
      }
    }
    conda_locations <- conda_locations[file.exists(conda_locations)]
    if (length(conda_locations) > 0)
      conda_locations
    else
      NULL
  } else {
    NULL
  }
}





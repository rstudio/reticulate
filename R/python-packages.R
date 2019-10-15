
#' Configure a Python Environment
#' 
#' Configure a Python environment, satisfying the Python dependencies of any
#' loaded \R packages.
#' 
#' Normally, this function should only be used by package authors, who want
#' to ensure that their package dependencies are installed in the active
#' Python environment. For example:
#' 
#' ```
#' .onLoad <- function(libname, pkgname) {
#'   reticulate::configure_environment()
#' }
#' ```
#' 
#' If the Python session has not yet been initialized, or if the user is not
#' using the default Miniconda Python installation, no action will be taken.
#' Otherwise, `reticulate` will take this as a signal to install any required
#' Python dependencies into the user's Python environment.
#' 
#' Note that, in the case where the Python session has not yet been initialized,
#' `reticulate` will automatically ensure your required Python dependencies
#' are installed after the Python session is initialized (when appropriate).
#' 
#' @export
configure_environment <- function(package = NULL) {
  
  if (!is_python_initialized())
    return(FALSE)
  
  # only done if we're using miniconda for now
  config <- py_config()
  python <- config$python
  home <- miniconda_path()
  if (substring(config$python, 1, nchar(home)) != home)
    return(FALSE)

  # find Python requirements  
  reqs <- python_package_requirements(package)
  if (length(reqs) == 0)
    return(FALSE)
  
  # get package requirements
  pkgreqs <- unlist(lapply(reqs, `[[`, "packages"), recursive = FALSE, use.names = FALSE)
  packages <- vapply(pkgreqs, `[[`, "package", FUN.VALUE = character(1))
  
  pip <- vapply(pkgreqs, function(req) {
    as.logical(req[["pip"]] %||% FALSE)
  }, FUN.VALUE = logical(1))
  
  # collect packages required from pip, conda
  pip_requested_packages <- packages[pip]
  if (length(pip_requested_packages)) {
    pip_installed_packages <- pip_freeze(python = python)
    pip_requested_packages <- setdiff(
      pip_requested_packages,
      tolower(pip_installed_packages$package)
    )
  }
  
  conda_requested_packages <- packages[!pip]
  if (length(conda_requested_packages)) {
    
    envpath <- dirname(dirname(python))
    conda <- miniconda_conda()
    conda_installed_packages <- conda_list_packages(
      envname = envpath,
      conda = conda
    )
    
    conda_requested_packages <- setdiff(
      conda_requested_packages,
      tolower(conda_installed_packages$package)
    )
    
  }
  
  conda_packages <- conda_requested_packages
  pip_packages   <- pip_requested_packages
  if (length(pip_packages) || length(conda_packages)) {
    
    if (length(pip_packages))
      py_install(pip_packages, pip = TRUE)
    
    if (length(conda_packages))
      py_install(conda_packages, pip = FALSE)
    
  }
  
  
  TRUE
}

python_package_requirements <- function(packages = NULL) {
  
  packages <- packages %||% loadedNamespaces()
  names(packages) <- packages
  reqs <- lapply(packages, function(package) {
    tryCatch(
      python_package_requirements_find(package),
      error = function(e) { warning(e); NULL }
    )
  })
  
  Filter(Negate(is.null), reqs)
  
}

python_package_requirements_find <- function(package) {
  
  descpath <- system.file("DESCRIPTION", package = package)
  desc <- read.dcf(descpath, all = TRUE)
  
  entry <- desc[["reticulate@R"]]
  if (is.null(entry))
    return(NULL)
  
  eval(parse(text = entry), envir = baseenv())
  
}

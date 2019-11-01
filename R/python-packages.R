
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
#'   reticulate::configure_environment(pkgname)
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
#' @param package The name of a package to configure. When `NULL`, `reticulate`
#'   will instead look at all loaded packages and discover their associated
#'   Python requirements.
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
  
  pkgreqs <- unlist(
    lapply(reqs, `[[`, "packages"),
    recursive = FALSE,
    use.names = FALSE
  )
  
  pip_installed_packages <- NULL
  conda_installed_packages <- NULL
  
  pip_packages <- character()
  conda_packages <- character()
  
  for (req in pkgreqs) {
    
    pip <- req$pip %||% TRUE
    components <- c(req$package, req$version)
    
    if (pip) {
      
      # read installed packages lazily
      if (is.null(pip_installed_packages)) {
        pip_installed_packages <- pip_freeze(python = python)
      }
      
      # construct requirement string
      requirement <- paste(components, collapse = "==")
      
      # check to see if we satisfy this requirement already
      satisfied <-
        requirement %in% pip_installed_packages$requirement %||%
        requirement %in% pip_installed_packages$package
      
      if (satisfied)
        next
        
      pip_packages[[length(pip_packages) + 1]] <- requirement
      
    } else {
      
      # read installed packages lazily
      envpath <- dirname(dirname(python))
      conda <- miniconda_conda()
      
      if (is.null(conda_installed_packages)) {
        conda_installed_packages <- conda_list_packages(
          envname = envpath,
          conda = conda
        )
      }
      
      # construct requirement string
      requirement <- paste(components, collapse = "=")
      
      # check to see if we satisfy this requirement already
      satisfied <-
        requirement %in% conda_installed_packages$requirement %||%
        requirement %in% conda_installed_packages$package
      
      if (satisfied)
        next
      
      conda_packages[[length(conda_packages + 1)]] <- requirement
    
    }
    
  }
  
  if (length(pip_packages))
      py_install(pip_packages, pip = TRUE)
    
  if (length(conda_packages))
    py_install(conda_packages, pip = FALSE)
  
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

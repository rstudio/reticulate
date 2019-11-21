
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
#' If you'd like to disable `reticulate`'s auto-configure behavior altogether,
#' you can set the environment variable:
#' 
#' ```
#' RETICULATE_AUTOCONFIGURE = FALSE
#' ```
#' 
#' e.g. in your `~/.Renviron` or similar.
#' 
#' Note that, in the case where the Python session has not yet been initialized,
#' `reticulate` will automatically ensure your required Python dependencies
#' are installed after the Python session is initialized (when appropriate).
#' 
#' @param package The name of a package to configure. When `NULL`, `reticulate`
#'   will instead look at all loaded packages and discover their associated
#'   Python requirements.
#'
#' @param force Boolean; force configuration of the associated environment?
#'
#' @export
configure_environment <- function(package = NULL, force = TRUE) {
  
  if (!is_python_initialized())
    return(FALSE)
  
  # allow users to opt out
  enabled <- Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = "TRUE")
  if (enabled %in% c("FALSE", "False", "0"))
    return(FALSE)
  
  # only done if we're using miniconda for now
  config <- py_config()
  python <- config$python
  home <- miniconda_path()
  if (!force && substring(config$python, 1, nchar(home)) != home)
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
  
  # check for incompatible package requests
  df <- do.call(rbind.data.frame, pkgreqs)
  splat <- split(df, df$package)
  pkgreqs <- enumerate(splat, function(pkg, requests) {
    
    rownames(requests) <- NULL
    
    # check for explicit requests by version
    explicit <- requests[!is.na(requests$version), ]
    if (nrow(explicit) == 0)
      return(requests[1, ])
    
    # check for single explicit version request
    n <- length(unique(requests$version))
    if (n == 1)
      return(explicit[1, ])
    
    # otherwise warn and sort
    explicit <- explicit[order(explicit$version, decreasing = TRUE), ]
    output <- capture.output(format(explicit))
    
    fmt <- "WARNING: incompatible requirements for package '%s' detected!"
    messagef(fmt, pkg)
    
    output <- capture.output(format(requests))
    message(paste(output, collapse = "\n"))
    selected <- explicit[1, ]
    
    fmt <- "WARNING: %s [%s] will be used."
    messagef(fmt, selected$package, selected$version)
    selected
    
  })
  
  # split into packages to be installed with pip vs. conda
  # we'll diff the requested packages against the currently-installed
  # packages and only install packages which truly need to be updated
  pip_installed_packages <- NULL
  conda_installed_packages <- NULL
  
  pip_packages <- character()
  conda_packages <- character()
  
  for (req in pkgreqs) {
    
    pip <- req$pip
    if (is.null(pip) || is.na(pip))
      pip <- TRUE
    
    version <- req$version
    if (is.null(version) || is.na(version))
      version <- NULL
    
    components <- c(req$package, version)
    
    if (pip) {
      
      # read installed packages lazily
      if (is.null(pip_installed_packages)) {
        pip_installed_packages <- pip_freeze(python = python)
      }
      
      # construct requirement string
      requirement <- paste(components, collapse = "==")
      
      # check to see if we satisfy this requirement already
      satisfied <-
        requirement %in% pip_installed_packages$requirement ||
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
        requirement %in% conda_installed_packages$requirement ||
        requirement %in% conda_installed_packages$package
      
      if (satisfied)
        next
      
      conda_packages[[length(conda_packages + 1)]] <- requirement
    
    }
    
  }
  
  if (length(pip_packages) || length(conda_packages)) {
    
    fmt <- "Configuring package '%s': please wait ..."
    messagef(fmt, package)
    
    if (length(pip_packages))
      py_install(pip_packages, pip = TRUE)
    
    if (length(conda_packages))
      py_install(conda_packages, pip = FALSE)
    
    message("Done!")
    
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
  
  spec <- eval(parse(text = entry), envir = baseenv())
  
  fields <- c("package", "version", "pip")
  spec$packages <- lapply(spec$packages, function(req) {
    
    if (is.null(req$package)) {
      warning("invalid spec provided by package '%s'", package)
      return(NULL)
    }
    
    data.frame(
      source  = package,
      package = as.character(req[["package"]]),
      version = as.character(req[["version"]] %||% NA),
      pip     = as.logical(req[["pip"]] %||% NA),
      stringsAsFactors = FALSE
    )
    
  })
  
  spec
  
}

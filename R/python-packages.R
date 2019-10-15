
configure_environment <- function() {
  
  # only done if we're using miniconda for now
  config <- py_config()
  home <- miniconda_path()
  if (substring(config$python, 1, nchar(home)) != home)
    return(FALSE)

  # find Python requirements  
  reqs <- python_package_requirements()
  
  # get package requirements
  pkgreqs <- if (is.list(reqs[["packages"]]))
    unlist(reqs[["packages"]], recursive = FALSE, use.names = FALSE)
  else
    reqs[["packages"]]
    
  packages <- vapply(pkgreqs, `[[`, "package", FUN.VALUE = character(1))
  pip <- vapply(pkgreqs, function(req) req[["pip"]] %||% FALSE, FUN.VALUE = logical(1))
  
  # list installed modules
  modules <- python_packages(python = config$python)
  
  # install them if necessary
  # TODO: we should compare requested versions and upgrade as needed
  pip_packages <- setdiff(packages[pip], modules$module)
  nonpip_packages <- setdiff(packages[!pip], modules$module)
  
  if (length(pip_packages) || length(nonpip_packages)) {
    
    message("One or more Python packages need to be installed -- please wait ...")
    
    if (length(pip_packages))
      py_install(pip_packages, pip = TRUE)
    
    if (length(nonpip_packages))
      py_install(nonpip_packages, pip = FALSE)
    
    message("Done!")
    
  }
  
  
  TRUE
}

python_package_requirements <- function() {
  
  packages <- loadedNamespaces()
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



#' Python configuration
#'
#' Information on Python and Numpy versions detected
#'
#' @return Python configuration object; Logical indicating whether Python
#'   bindings are available
#'
#' @export
py_config <- function() {
  ensure_python_initialized()
  .globals$py_config
}


#' Check if Python is available on this system
#' 
#' @param initialize `TRUE` to attempt to initialize Python bindings if they
#'   aren't yet available (defaults to `FALSE`).
#'   
#' @return Logical indicating whether Python is initialized.
#'   
#' @note The `py_numpy_available` function is a superset of the
#'  `py_available` function (it calls `py_available` first before
#'  checking for NumPy).   
#'   
#' @export
py_available <- function(initialize = FALSE) {
  if (is_python_initialized())
    .globals$py_config$available
  else if (initialize) {
    tryCatch({
      ensure_python_initialized()
      .globals$py_config$available
    }, error = function(e) FALSE)
  } else {
    FALSE
  }
}


#' @rdname py_available
#' @export
py_numpy_available <- function(initialize = FALSE) {
  if (!py_available(initialize = initialize))
    FALSE
  else
    py_numpy_available_impl()
}


#' Check if a Python module is available on this system.
#' 
#' @param module Name of module
#'   
#' @return Logical indicating whether module is available
#'   
#' @export
py_module_available <- function(module) {
  tryCatch({ import(module); TRUE }, error = function(e) FALSE)
}


#' Discover the version of Python to use with reticulate.
#' 
#' This function enables callers to check which versions of Python will
#' be discovered on a system as well as which one will be chosen for 
#' use with reticulate.
#' 
#' @param required_module A optional module name that must be available
#'   in order for a version of Python to be used. 
#' 
#' @return Python configuration object.
#' 
#' @export
py_discover_config <- function(required_module = NULL) {

  # create a list of possible python versions to bind to
  python_versions <- reticulate_python_versions()

  # look on system path
  python <- Sys.which("python")
  if (nzchar(python))
    python_versions <- c(python_versions, python)

  # provide other common locations
  if (is_windows()) {
    extra_versions <- windows_registry_python_versions(required_module)
  } else {
    extra_versions <- c(
      "/usr/bin/python",
      "/usr/local/bin/python",
      "/opt/python/bin/python",
      "/opt/local/python/bin/python",
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/opt/python/bin/python3",
      "/opt/local/python/bin/python3",
      path.expand("~/anaconda/bin/python"),
      path.expand("~/anaconda3/bin/python")
    )

    # if we have a required module then hunt for virtualenvs or condaenvs that
    # share it's name as well
    if (!is.null(required_module)) {
      extra_versions <- c(
        path.expand(sprintf("~/%s/bin/python", required_module)),
        extra_versions,
        path.expand(sprintf("~/anaconda/envs/%s/bin/python", required_module)),
        path.expand(sprintf("~/anaconda3/envs/%s/bin/python", required_module))
      )
    }
  }

  # filter locations by existence
  python_versions <- unique(c(python_versions, extra_versions))
  python_versions <- python_versions[file.exists(python_versions)]

  # scan until we find a version of python that meets our qualifying conditions
  valid_python_versions <- c()
  for (python_version in python_versions) {

    # get the config
    config <- python_config(python_version, required_module, python_versions)

    # if we have a required module ensure it's satsified.
    # also check architecture (can be an issue on windows)
    has_compatible_arch = !is_incompatible_arch(config)
    has_preferred_numpy <- !is.null(config$numpy) && config$numpy$version >= "1.6"
    if (has_compatible_arch && has_preferred_numpy)
      valid_python_versions <- c(valid_python_versions, python_version)
    has_required_module <- is.null(config$required_module) || !is.null(config$required_module_path)
    if (has_compatible_arch && has_preferred_numpy && has_required_module) 
      return(config)
  }

  # no preferred found, return first with valid config if we have it or NULL
  if (length(valid_python_versions) > 0)
    return(python_config(valid_python_versions[[1]], required_module, python_versions))
  else if (length(python_versions) > 0)
    return(python_config(python_versions[[1]], required_module, python_versions))  
  else
    return(NULL)
}


python_config <- function(python, required_module, python_versions) {

  # collect configuration information
  if (!is.null(required_module)) {
    Sys.setenv(RETICULATE_REQUIRED_MODULE = required_module)
    on.exit(Sys.unsetenv("RETICULATE_REQUIRED_MODULE"), add = TRUE)
  }
  config_script <- system.file("config/config.py", package = "reticulate")
  config <- system2(command = python, args = paste0('"', config_script, '"'), stdout = TRUE)
  status <- attr(config, "status")
  if (!is.null(status)) {
    errmsg <- attr(config, "errmsg")
    stop("Error ", status, " occurred running ", python, " ", errmsg)
  }

  config <- read.dcf(textConnection(config), all = TRUE)

  # get the full textual version and the numeric version, check for anaconda
  version_string <- config$Version
  version <- config$VersionNumber
  anaconda <- grepl("continuum", tolower(version_string)) || grepl("anaconda", tolower(version_string))
  architecture <- config$Architecture

  # determine the location of libpython (see also # https://github.com/JuliaPy/PyCall.jl/blob/master/deps/build.jl)
  if (is_windows()) {
    # note that 'prefix' has the binary location and 'py_version_nodot` has the suffix`
    python_libdir <- dirname(python)
    python_dll <- paste0("python", gsub(".", "", version, fixed = TRUE), ".dll")
    libpython <- file.path(python_libdir, python_dll)
    if (!file.exists(libpython))
      libpython <- python_dll
  } else {
    # (note that the LIBRARY variable has the name of the static library)
    python_libdir_config <- function(var) {
      python_libdir <- config[[var]]
      ext <- switch(Sys.info()[["sysname"]], Darwin = ".dylib", Windows = ".dll", ".so")
      libpython <- file.path(python_libdir, paste0("libpython" , version, c("", "m"), ext))
      libpython_exists <- libpython[file.exists(libpython)]
      if (length(libpython_exists) > 0)
        libpython_exists[[1]]
      else
        libpython[[1]]
    }
    libpython <- python_libdir_config("LIBPL")
    if (!file.exists(libpython))
      libpython <- python_libdir_config("LIBDIR")
  }

  # determine PYTHONHOME
  pythonhome <- config$PREFIX
  if (!is_windows())
    pythonhome <- paste(pythonhome, config$EXEC_PREFIX, sep = ":")


  as_numeric_version <- function(version) {
    version <- clean_version(version)
    numeric_version(version)
  }

  # check for numpy
  if (!is.null(config$NumpyPath))
    numpy <- list(path = config$NumpyPath,
                  version = as_numeric_version(config$NumpyVersion))
  else
    numpy <- NULL

  # check for virtualenv activate script
  activate_this <- file.path(dirname(python), "activate_this.py")
  if (file.exists(activate_this))
    virtualenv_activate <- activate_this
  else
    virtualenv_activate <- ""

  # check for required module
  required_module_path <- config$RequiredModulePath

  # return config info
  structure(class = "py_config", list(
    python = python,
    libpython = libpython,
    pythonhome = pythonhome,
    virtualenv_activate = virtualenv_activate,
    version_string = version_string,
    version = version,
    architecture = architecture,
    anaconda = anaconda,
    numpy = numpy,
    required_module = required_module,
    required_module_path = required_module_path,
    available = FALSE,
    python_versions = python_versions
  ))

}

#' @export
str.py_config <- function(object, ...) {
  x <- object
  out <- ""
  out <- paste0(out, "python:         ", x$python, "\n")
  out <- paste0(out, "libpython:      ", x$libpython, ifelse(is_windows() || file.exists(x$libpython), "", "[NOT FOUND]"), "\n")
  out <- paste0(out, "pythonhome:     ", x$pythonhome, "\n")
  if (nzchar(x$virtualenv_activate))
    out <- paste0(out, "virtualenv:     ", x$virtualenv_activate, "\n")
  out <- paste0(out, "version:        ", x$version_string, "\n")
  if (is_windows())
    out <- paste0(out, "Architecture:   ", x$architecture, "\n")
  if (!is.null(x$numpy)) {
    out <- paste0(out, "numpy:          ", x$numpy$path, "\n")
    out <- paste0(out, "numpy_version:  ", as.character(x$numpy$version), "\n")
  } else {
    out <- paste0(out, "numpy:           [NOT FOUND]\n")
  }
  if (!is.null(x$required_module)) {
    out <- paste0(out, sprintf("%-16s", paste0(x$required_module, ":")))
    if (!is.null(x$required_module_path))
      out <- paste0(out, x$required_module_path, "\n")
    else
      out <- paste0(out, "[NOT FOUND]\n")
  }
  if (length(x$python_versions) > 1) {
    out <- paste0(out, "\npython versions found: \n")
    python_versions <- paste0(" ", x$python_versions, collapse = "\n")
    out <- paste0(out, python_versions, sep = "\n")
  }
  out
}

#' @export
print.py_config <- function(x, ...) {
 cat(str(x))
}


is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}


clean_version <- function(version) {
  gsub("\\.$", "", gsub("[A-Za-z_]+", "", version))
}

reticulate_python_versions <- function() {

  # python versions to return
  python_versions <- c()
  
  # combine registered versions with the RETICULATE_PYTHON environment variable
  reticulate_python_options <- .globals$use_python_versions
  reticulate_python_env <- Sys.getenv("RETICULATE_PYTHON", unset = NA)
  if (!is.na(reticulate_python_env))
    reticulate_python_options <- c(reticulate_python_env, reticulate_python_options)
                                 
  
  # determine python versions to return
  if (length(reticulate_python_options) > 0) {
    for (i in 1:length(reticulate_python_options)) {
      python <- normalize_python_path(reticulate_python_options[[i]])
      if (python$exists)
        python_versions <- c(python_versions, python$path)
    }
  }
  
  # return them
  python_versions
}


normalize_python_path <- function(python) {
  
  # normalize trailing slash and expand
  python <- gsub("[\\/]+$", "", python)
  python <- path.expand(python)
  
  # check for existence
  if (!utils::file_test("-d", python) &&
      !utils::file_test("-f", python)) {
    list(
      path = python,
      exists = FALSE
    )
  } else {
    
    # append binary if it's a directory
    if (utils::file_test("-d", python))
      python <- file.path(python, "python")
    
    # append .exe if necessary on windows
    if (is_windows() && (!grepl("^.*\\.exe$", tolower(python))))
      python <- paste0(python, ".exe")
    
    # return
    list(
      path = python,
      exists = TRUE
    )
  }
  
}


windows_registry_python_versions <- function(required_module) {

  

  python_core_versions <- c(read_python_versions_from_registry("HCU", key = "PythonCore"),
                            read_python_versions_from_registry("HLM", key = "PythonCore"))


  anaconda_versions <- read_anaconda_versions_from_registry()
  if (!is.null(required_module) && length(anaconda_versions) > 0) {
    anaconda_envs <- utils::shortPathName(
      file.path(dirname(anaconda_versions), "envs", required_module, "python.exe")
    )
  } else {
    anaconda_envs <- NULL
  }

  c(python_core_versions, anaconda_envs, anaconda_versions)
}

read_anaconda_versions_from_registry <- function() {
  c(read_python_versions_from_registry("HCU", key = "ContinuumAnalytics"),
    read_python_versions_from_registry("HLM", key = "ContinuumAnalytics"))
}

read_python_versions_from_registry <- function(hive,key) {
  versions <- c()
  python_core_key <- tryCatch(utils::readRegistry(
    key = paste0("SOFTWARE\\Python\\", key), hive = hive, maxdepth = 3),
    error = function(e) NULL)
  
  if (length(python_core_key) > 0) {
    for (version in names(python_core_key)) {
      version_key <- python_core_key[[version]]
      if (is.list(version_key) && !is.null(version_key$InstallPath)) {
        version_dir <- version_key$InstallPath$`(Default)`
        version_dir <- gsub("[\\/]+$", "", version_dir)
        version_exe <- paste0(version_dir, "\\python.exe")
        versions <- c(versions, utils::shortPathName(version_exe))
      }
    }
  }
  
  versions
}

# convert R arch to python arch
python_arch <- function() {
  if (.Platform$r_arch == "i386")
    "32bit"
  else if (.Platform$r_arch == "x64")
    "64bit"
  else
    "Unknown"
}


# check for compatible architecture
is_incompatible_arch <- function(config) {
  if (is_windows()) {
    !identical(python_arch(),config$architecture)
  } else {
    FALSE
  }
}


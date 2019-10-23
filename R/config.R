

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


#' Build Python configuration error message
#'
#' @param prefix Error message prefix
#'
#' @keywords internal
#' @export
py_config_error_message <- function(prefix) {
  message <- prefix
  config <- py_config()
  if (!is.null(config)) {
    message <- paste0(message, "\n\nDetected Python configuration:\n\n",
                      str(config), "\n")
  }
  message
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
  tryCatch({ import(module); TRUE }, error = clear_error_handler(FALSE))
}


#' Discover the version of Python to use with reticulate.
#'
#' This function enables callers to check which versions of Python will
#' be discovered on a system as well as which one will be chosen for
#' use with reticulate.
#'
#' @param required_module A optional module name that must be available
#'   in order for a version of Python to be used.
#' @param use_environment An optional virtual/conda environment name
#'   to prefer in the search
#'
#' @return Python configuration object.
#'
#' @export
py_discover_config <- function(required_module = NULL, use_environment = NULL) {

  # if PYTHON_SESSION_INITIALIZED is specified then use it without scanning
  # further (this is a "hard" requirement because an embedding process may
  # set this to indicate that the python interpreter is already loaded)
  py_session_initialized <- py_session_initialized_binary()
  if (!is.null(py_session_initialized)) {
    python_version <- normalize_python_path(py_session_initialized)$path
    config <- python_config(python_version, required_module, python_version, forced = "PYTHON_SESSION_INITIALIZED")
    return(config)
  }

  # if RETICULATE_PYTHON is specified then use it without scanning further
  reticulate_env <- Sys.getenv("RETICULATE_PYTHON", unset = NA)
  if (!is.na(reticulate_env)) {
    python_version <- normalize_python_path(reticulate_env)
    if (!python_version$exists)
      stop("Python specified in RETICULATE_PYTHON (", reticulate_env, ") does not exist")
    python_version <- python_version$path
    config <- python_config(python_version, required_module, python_version, forced = "RETICULATE_PYTHON")
    return(config)
  }

  # if RETICULATE_PYTHON_ENV is specified then use that
  reticulate_python_env <- Sys.getenv("RETICULATE_PYTHON_ENV", unset = NA)
  if (!is.na(reticulate_python_env)) {
    python <- python_binary_path(reticulate_python_env)
    python_version <- normalize_python_path(python)
    if (!python_version$exists)
      stop("Python specified in RETICULATE_PYTHON_ENV (", reticulate_python_env, ") does not exist")
    path <- python_version$path
    config <- python_config(path, required_module, path, forced = "RETICULATE_PYTHON_ENV")
    return(config)
  }

  # next look for a required python version (e.g. use_python("/usr/bin/python", required = TRUE))
  required_version <- .globals$required_python_version
  if (!is.null(required_version)) {
    python_version <- normalize_python_path(required_version)$path
    config <- python_config(python_version, required_module, python_version, forced = "use_python function")
    return(config)
  }

  # create a list of possible python versions to bind to
  # (start with versions specified via environment variable or use_* function)
  python_versions <- reticulate_python_versions()

  # prioritize the r-reticulate python environment
  python_virtualenvs <- python_virtualenv_versions()
  r_reticulate_python_envs <- python_virtualenvs[python_virtualenvs$name == "r-reticulate", ]
  python_versions <- c(python_versions, r_reticulate_python_envs$python)
  
  # next look in virtual environments that have a required module derived name
  if (!is.null(required_module)) {
    # filter by required module
    envnames <- c(required_module, paste0("r-", required_module), use_environment)
    module_python_envs <- python_virtualenvs[python_virtualenvs$name %in% envnames, ]
    python_versions <- c(python_versions, module_python_envs$python)
  }

  # look for r-reticulate environment in miniconda
  # if the environment doesn't exist, and the user hasn't requested a separate
  # environment, then we'll prompt for installation of miniconda
  miniconda <- miniconda_conda()
  if (!file.exists(miniconda)) {
    
    can_install_miniconda <-
      interactive() &&
      length(python_versions) == 0 &&
      miniconda_installable()
    
    if (can_install_miniconda)
      miniconda_install_prompt()
    
  }
  
  # if the earlier branch installed miniconda, it may exist now -- if so,
  # try to activate it
  if (file.exists(miniconda)) {
    
    # create the conda environment if necessary
    envpath <- miniconda_python_envpath()
    if (!file.exists(envpath)) {
      python <- miniconda_python_package()
      conda_create(envpath, packages = c(python, "numpy"), conda = miniconda)
    }
    
    # bind to it
    miniconda_python <- conda_python(envpath, conda = miniconda)
    config <- python_config(miniconda_python, NULL, miniconda_python)
    return(config)
    
  }
  
  # look for conda environments
  python_condaenvs <- python_conda_versions()
  r_reticulate_python_envs <- python_condaenvs[python_condaenvs$name == "r-reticulate", ]
  python_versions <- c(python_versions, r_reticulate_python_envs$python)
  
  # next look in virtual environments that have a required module derived name
  if (!is.null(required_module)) {
    # filter by required module
    envnames <- c(required_module, paste0("r-", required_module), use_environment)
    module_python_envs <- python_condaenvs[python_condaenvs$name %in% envnames, ]
    python_versions <- c(python_versions, module_python_envs$python)
  }

  # join virtualenv, condaenv environments together
  python_envs <- rbind(python_virtualenvs, python_condaenvs)
  
  # look on system path
  python <- as.character(Sys.which("python"))
  if (nzchar(python))
    python_versions <- c(python_versions, python)

  # provide other common locations
  if (is_windows()) {
    python_versions <- c(python_versions, py_versions_windows()$executable_path)
  } else {
    python_versions <- c(python_versions,
                         "/usr/bin/python3",
                         "/usr/local/bin/python3",
                         "/opt/python/bin/python3",
                         "/opt/local/python/bin/python3",
                         "/usr/bin/python",
                         "/usr/local/bin/python",
                         "/opt/python/bin/python",
                         "/opt/local/python/bin/python",
                         path.expand("~/anaconda3/bin/python")
                         path.expand("~/anaconda/bin/python"),
    )
  }

  # next add all known virtual environments
  python_versions <- c(python_versions, python_envs$python)

  # de-duplicate
  python_versions <- unique(python_versions)

  # filter locations by existence
  if (length(python_versions) > 0)
    python_versions <- python_versions[file.exists(python_versions)]

  # remove 'fake' / inaccessible python executables
  # https://github.com/rstudio/reticulate/issues/534
  if (is_windows()) {
    info <- suppressWarnings(file.info(python_versions))
    size <- ifelse(is.na(info$size), 0, info$size)
    python_versions <- python_versions[size != 0]
  }

  # scan until we find a version of python that meets our qualifying conditions
  valid_python_versions <- c()
  for (python_version in python_versions) {

    # get the config
    config <- python_config(python_version, required_module, python_versions)

    # if we have a required module ensure it's satisfied.
    # also check architecture (can be an issue on windows)
    has_python_gte_27 <- as.numeric_version(config$version) >= "2.7"
    has_compatible_arch <- !is_incompatible_arch(config)
    has_preferred_numpy <- !is.null(config$numpy) && config$numpy$version >= "1.6"
    if (has_compatible_arch && has_preferred_numpy)
      valid_python_versions <- c(valid_python_versions, python_version)
    has_required_module <- is.null(config$required_module) || !is.null(config$required_module_path)
    if (has_python_gte_27 && has_compatible_arch && has_preferred_numpy && has_required_module)
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


#' Discover versions of Python installed on a Windows system
#'
#' @return Data frame with `type`, `hive`, `install_path`, `executable_path`,
#'   and `version`.
#'
#' @keywords internal
#' @export
py_versions_windows <- function() {
  rbind(
    read_python_versions_from_registry("HCU", key = "PythonCore"),
    read_python_versions_from_registry("HLM", key = "PythonCore"),
    windows_registry_anaconda_versions()
  )
}

python_virtualenv_versions <- function() {
  home <- virtualenv_root()
  bins <- python_environments(home)
  data.frame(
    name = basename(dirname(dirname(bins))),
    python = bins,
    stringsAsFactors = FALSE
  )
}

python_conda_versions <- function() {
  
  if (is_windows()) {
    
    # list all conda environments
    conda_envs <- data.frame(name = character(),
                             python = character(),
                             stringsAsFactors = FALSE)
    registry_versions <- py_versions_windows()
    anaconda_registry_versions <- subset(registry_versions, registry_versions$type == "Anaconda")
    for (conda in file.path(anaconda_registry_versions$install_path, "Scripts", "conda.exe")) {
      conda_envs <- rbind(conda_envs, conda_list(conda = conda))
    }
    
    conda_envs
    
  } else {
    
    env_dirs <- c("~/anaconda/envs",
                  "~/anaconda2/envs",
                  "~/anaconda3/envs",
                  "~/anaconda4/envs",
                  "~/miniconda/envs",
                  "~/miniconda2/envs",
                  "~/miniconda3/envs",
                  "~/miniconda4/envs",
                  "/anaconda/envs",
                  "/anaconda2/envs",
                  "/anaconda3/envs",
                  "/anaconda4/envs",
                  "/miniconda/envs",
                  "/miniconda2/envs",
                  "/miniconda3/envs",
                  "/miniconda4/envs",
                  "~")
    
    python_env_binaries <- python_environments(env_dirs)
    data.frame(name = basename(dirname(dirname(python_env_binaries))),
               python = python_env_binaries,
               stringsAsFactors = FALSE)
  }
  
}

python_environments <- function(env_dirs, required_module = NULL) {

  # filter env_dirs by existence
  env_dirs <- env_dirs[utils::file_test("-d", env_dirs)]

  # envs to return
  envs <- character()

  # python bin differs by platform
  python_bin <- ifelse(is_windows(), "python.exe", "bin/python")

  for (env_dir in env_dirs) {
    # filter by required module if requested
    if (!is.null(required_module)) {
      module_envs <- c(paste0("r-", required_module), required_module)
      envs <- c(envs, path.expand(sprintf("%s/%s/%s", env_dir, module_envs, python_bin)))

      # otherwise return all
    } else {
      envs <- c(envs, path.expand(sprintf("%s/%s",
                                          list.dirs(env_dir, recursive = FALSE),
                                          python_bin)))
    }
  }

  # filter by existence
  if (length(envs) > 0)
    envs[file.exists(envs)]
  else
    envs
}

python_munge_path <- function(python) {

  # add the python bin dir to the PATH (so that any execution of python from
  # within the interpreter, from a system call, or from within a terminal
  # hosted within the front end will use the same version of python.
  #
  # we do this up-front in python_config as otherwise attempts to discover
  # and load numpy can fail, especially on Windows
  # https://github.com/rstudio/reticulate/issues/367
  python_home <- dirname(python)
  python_dirs <- c(normalizePath(python_home))
  if (is_windows()) {

    # include the Scripts path, as well
    python_scripts <- file.path(python_home, "Scripts")
    if (file.exists(python_scripts))
      python_dirs <- c(python_dirs, normalizePath(python_scripts))

    # we saw some crashes occurring when Python modules attempted to load
    # dynamic libraries at runtime; e.g.
    #
    #   Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll
    #
    # we work around this by putting the associated binary directory
    # on the PATH so it can be successfully resolved
    python_library_bin <- file.path(python_home, "Library/bin")
    if (file.exists(python_library_bin))
      python_dirs <- c(python_dirs, normalizePath(python_library_bin))
  }

  path_prepend(python_dirs)

}

python_config <- function(python, required_module, python_versions, forced = NULL) {

  # normalize and remove duplicates
  python <- canonical_path(python)
  python_versions <- canonical_path(python_versions)
  python_versions <- unique(python_versions)
  
  # update and restore PATH when done
  oldpath <- python_munge_path(python)
  on.exit(Sys.setenv(PATH = oldpath), add = TRUE)

  # collect configuration information
  if (!is.null(required_module)) {
    Sys.setenv(RETICULATE_REQUIRED_MODULE = required_module)
    on.exit(Sys.unsetenv("RETICULATE_REQUIRED_MODULE"), add = TRUE)
  }

  config_script <- system.file("config/config.py", package = "reticulate")
  # It seems that Windows in R3.6 returns the configuration from the python file only in stderr and not in stdout
  config <- system2(command = python, args = paste0('"', config_script, '"'), stdout = TRUE, stderr = FALSE)
  status <- attr(config, "status")
  if (!is.null(status)) {
    errmsg <- attr(config, "errmsg")
    stop("Error ", status, " occurred running ", python, " ", errmsg)
  }

  config_connection <- textConnection(config)
  on.exit(close(config_connection))
  config <- read.dcf(config_connection, all = TRUE)

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
      pattern <- paste0("^libpython", version, "d?m?", ext)
      libpython <- list.files(python_libdir, pattern = pattern, full.names = TRUE)
    }
    for (libsrc in c("LIBPL", "LIBDIR")) {
      if (!is.null(config[[libsrc]])) {
        libpython <- python_libdir_config(libsrc)
        if (length(libpython))
          break
        else
          libpython <- NULL
      } else {
        libpython <- NULL
      }
    }
  }

  # determine PYTHONHOME
  pythonhome <- NULL
  if (!is.null(config$PREFIX)) {
    pythonhome <- canonical_path(config$PREFIX)
    if (!is_windows()) {
      exec_prefix <- canonical_path(config$EXEC_PREFIX)
      pythonhome <- paste(pythonhome, exec_prefix, sep = ":")
    }
  }


  as_numeric_version <- function(version) {
    version <- clean_version(version)
    numeric_version(version)
  }

  # check for numpy
  numpy <- NULL
  if (!is.null(config$NumpyPath)) {
    numpy <- list(
      path = canonical_path(config$NumpyPath),
      version = as_numeric_version(config$NumpyVersion)
    )
  }

  # check to see if this is a Python virtualenv
  root <- dirname(dirname(python))
  virtualenv <- if (is_virtualenv(root))
    root
  else
    ""

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
    libpython = libpython[1],
    pythonhome = pythonhome,
    virtualenv = virtualenv,
    virtualenv_activate = virtualenv_activate,
    version_string = version_string,
    version = version,
    architecture = architecture,
    anaconda = anaconda,
    numpy = numpy,
    required_module = required_module,
    required_module_path = required_module_path,
    available = FALSE,
    python_versions = python_versions,
    forced = forced
  ))

}

#' @export
str.py_config <- function(object, ...) {
  x <- object
  out <- ""
  out <- paste0(out, "python:         ", x$python, "\n")
  out <- paste0(out, "libpython:      ", ifelse(is.null(x$libpython), "[NOT FOUND]", x$libpython), ifelse(is_windows() || is.null(x$libpython) || file.exists(x$libpython), "", "[NOT FOUND]"), "\n")
  out <- paste0(out, "pythonhome:     ", ifelse(is.null(x$pythonhome), "[NOT FOUND]", x$pythonhome), "\n")
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
  if (!is.null(x$forced)) {
    out <- paste0(out, "\nNOTE: Python version was forced by ", x$forced, "\n")
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

is_unix <- function() {
  identical(.Platform$OS.type, "unix")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

is_linux <- function() {
  identical(tolower(Sys.info()[["sysname"]]), "linux")
}

is_ubuntu <- function() {
  # check /etc/lsb-release
  if (is_unix() && file.exists("/etc/lsb-release")) {
    lsbRelease <- readLines("/etc/lsb-release")
    any(grepl("Ubuntu", lsbRelease))
  } else {
    FALSE
  }
}

is_rstudio <- function() {
  exists("RStudio.Version", envir = globalenv())
}

is_rstudio_desktop <- function() {
  if (!exists("RStudio.Version", envir = globalenv()))
    return(FALSE)

  RStudio.Version <- get("RStudio.Version", envir = globalenv())
  version <- RStudio.Version()
  identical(version$mode, "desktop")
}

clean_version <- function(version) {
  gsub("\\.$", "", gsub("[A-Za-z_+].*$", "", version))
}

reticulate_python_versions <- function() {

  # python versions to return
  python_versions <- c()

  # get versions specified via use_* functions
  reticulate_python_options <- .globals$use_python_versions

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


windows_registry_anaconda_versions <- function() {
  rbind(read_python_versions_from_registry("HCU", key = "ContinuumAnalytics", type = "Anaconda"),
        read_python_versions_from_registry("HLM", key = "ContinuumAnalytics", type = "Anaconda"))
}

read_python_versions_from_registry <- function(hive, key,type=key) {

  python_core_key <- tryCatch(utils::readRegistry(
    key = paste0("SOFTWARE\\Python\\", key), hive = hive, maxdepth = 3),
    error = function(e) NULL)


  types <- c()
  hives <- c()
  install_paths <- c()
  executable_paths <- c()
  versions <- c()
  archs <- c()

  if (length(python_core_key) > 0) {
    for (version in names(python_core_key)) {
      version_key <- python_core_key[[version]]
      if (is.list(version_key) && !is.null(version_key$InstallPath)) {
        version_dir <- version_key$InstallPath$`(Default)`
        if (!is.null(version_dir) && utils::file_test("-d", version_dir)) {

          # determine install_path and executable_path
          install_path <- version_dir
          executable_path <- file.path(install_path, "python.exe")

          # proceed if it exists
          if (file.exists(executable_path)) {

            # determine version and arch
            if (type == "Anaconda") {
              matches <- regexec("^Anaconda.*(32|64).*$", version)
              matches <- regmatches(version, matches)[[1]]
              if (length(matches) == 2) {
                version <- version_key$SysVersion
                arch <- matches[[2]]
              } else {
                warning("Unexpected format for Anaconda version: ", version,
                        "\n(Please install a more recent version of Anaconda)")
                arch <- NA
              }
            } else { # type == "PythonCore"
              matches <- regexec("^(\\d)\\.(\\d)(?:-(32|64))?$", version)
              matches <- regmatches(version, matches)[[1]]
              if (length(matches) == 4) {
                version <- paste(matches[[2]], matches[[3]], sep = ".")
                arch <- matches[[4]]
                if (!nzchar(arch)) {
                  if (numeric_version(version) >= "3.0")
                    arch <- "64"
                  else {
                    python_arch <- python_arch(executable_path)
                    arch <- gsub("bit", "", python_arch, fixed = TRUE)
                  }
                }
              } else {
                warning("Unexpected format for PythonCore version: ", version)
                arch <- NA
              }
            }

            if (!is.na(arch)) {
              # convert to R arch
              if (arch == "32")
                arch <- "i386"
              else if (arch == "64")
                arch <- "x64"

              # append to vectors
              types <- c(types, type)
              hives <- c(hives, hive)
              install_paths <- c(install_paths, utils::shortPathName(install_path))
              executable_paths <- c(executable_paths, utils::shortPathName(executable_path))
              versions <- c(versions, version)
              archs <- c(archs, arch)
            }
          }
        }
      }
    }
  }

  data.frame(
    type = types,
    hive = hives,
    install_path = install_paths,
    executable_path = executable_paths,
    version = versions,
    arch = archs,
    stringsAsFactors = FALSE
  )
}


# get the architecture from a python binary
python_arch <- function(python) {

  # run command
  result <- system2(python, stdout = TRUE, args = c("-c", shQuote(
    "import sys; import platform; sys.stdout.write(platform.architecture()[0])")))

  # check for error
  error_status <- attr(result, "status")
  if (!is.null(error_status))
    stop("Error ", error_status, " occurred while checking for python architecture", call. = FALSE)

  # return arch
  result

}

# convert R arch to python arch
current_python_arch <- function() {
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
    !identical(current_python_arch(),config$architecture)
  } else {
    FALSE
  }
}


py_session_initialized_binary <- function() {

  # binary to return
  python_binary <- NULL

  # check environment variable
  py_session <- Sys.getenv("PYTHON_SESSION_INITIALIZED", unset = NA)
  if (!is.na(py_session)) {
    py_session <- strsplit(py_session, ":", fixed = TRUE)[[1]]
    py_session <- strsplit(py_session, "=", fixed = TRUE)
    keys <- character()
    py_session <- lapply(py_session, function(x) {
      keys <<- c(keys, x[[1]])
      x[[2]]
    })
    if (all(c("current_pid", "sys.executable") %in% keys)) {
      names(py_session) <- keys
      # verify it's from the current process
      if (identical(as.character(Sys.getpid()), py_session$current_pid)) {
        python_binary <- py_session$sys.executable
      }
    } else {
      warning("PYTHON_SESSION_INITIALIZED does not include current_pid and sys.executable",
              call. = FALSE)
    }
  }

  # return
  python_binary
}






# TODO: performance of tf_config scanning

tf_config <- function() {
  .tf_config
}


tf_discover_config <- function() {

  # create a list of possible python versions to bind to
  python_versions <- character()

  # look for environment variable
  tensorflow_python <- tensorflow_python()
  if (!is.null(tensorflow_python())) {
    if (tensorflow_python$exists)
      python_versions <- c(python_versions, tensorflow_python$python)
    else
      warning("Specified TENSORFLOW_PYTHON '", tensorflow_python$python, "' does not exist.")
  }

  # look on system path
  python <- Sys.which("python")
  if (nzchar(python))
    python_versions <- c(python_versions, python)

  # provide other common locations
  if (is_windows()) {
    extra_versions <- windows_registry_python_versions()
  } else {
    extra_versions <- c(
      path.expand("~/tensorflow/bin/python"),
      "/usr/local/bin/python",
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/opt/python/bin/python",
      "/opt/local/python/bin/python",
      "/opt/python/bin/python3",
      "/opt/local/python/bin/python3"
    )
  }
  python_versions <- c(python_versions, extra_versions)
  python_versions <- normalizePath(python_versions, mustWork = FALSE)
  python_versions <- unique(python_versions)

  # filter locations by existence
  python_versions <- python_versions[file.exists(python_versions)]

  # scan until we find a version of tensorflow
  for (python_version in python_versions) {
    config <- tf_python_config(python_version, python_versions)
    if (!is.null(config$tensorflow) && !config$anaconda)
      return(config)
  }

  # no version of tf found, return first if we have it or NULL
  if (length(python_versions) >= 1)
    return(tf_python_config(python_versions[[1]], python_versions))
  else
    return(NULL)
}


tf_python_config <- function(python, python_versions) {

  # helper to execute python code and return stdout
  exec_python <- function(command) {
    system(command = paste(shQuote(python), "-c", shQuote(command)), intern = TRUE)
  }

  py_config_var <- function(var) {
    exec_python(sprintf("import sys; import sysconfig; sys.stdout.write(sysconfig.get_config_vars('%s')[0]);",
                        var))
  }

  py_sys_var <- function(var) {
    exec_python(sprintf("import sys; sys.stdout.write(sys.%s);", var))
  }

  # get the full textual version and the numeric version, check for anaconda
  version_string <- py_sys_var("version")
  version <- exec_python("import sys; sys.stdout.write(str(sys.version_info.major) + '.' + str(sys.version_info.minor));")
  anaconda <- grepl("continuum", tolower(version_string)) || grepl("anaconda", tolower(version_string))

  # determine the location of libpython (see also # https://github.com/JuliaPy/PyCall.jl/blob/master/deps/build.jl)
  if (is_windows()) {
    # note that 'prefix' has the binary location and 'py_version_nodot` has the suffix`
    python_libdir <- dirname(python)
    libpython <- file.path(python_libdir, paste0("python", gsub(".", "", version, fixed = TRUE), ".dll"))
  } else {
    # (note that the LIBRARY variable has the name of the static library)
    python_libdir_config <- function(var) {
      python_libdir <- py_config_var(var)
      ext <- switch(Sys.info()[["sysname"]], Darwin = ".dylib", Windows = ".dll", ".so")
      libpython <- file.path(python_libdir, paste0("libpython" , version, ext))
    }
    libpython <- python_libdir_config("LIBPL")
    if (!file.exists(libpython))
      libpython <- python_libdir_config("LIBDIR")
  }

  # determine PYTHONHOME
  pythonhome <- normalizePath(py_sys_var("prefix"), mustWork = FALSE)
  if (!is_windows())
    pythonhome <- paste(pythonhome,
                        normalizePath(py_sys_var("exec_prefix"), mustWork = FALSE),
                        sep = ":")

  # function to check for a python module and it's version
  find_python_module <- function(module) {
    found <- !identical(exec_python(sprintf("import sys; import pkgutil; sys.stdout.write(str(pkgutil.find_loader('%s')));", module)), "None")
    if (found) {
      list(
        path = normalizePath(exec_python(sprintf("import sys; import %s; sys.stdout.write(%s.__path__[0]);", module, module))),
        version = numeric_version(clean_tf_version(exec_python(sprintf("import sys; import %s; sys.stdout.write(%s.__version__);", module, module))))
      )
    } else {
      NULL
    }
  }

  # check for numpy and tensorflow
  numpy <- find_python_module("numpy")
  tensorflow <- find_python_module("tensorflow")

  # return config info
  structure(class = "tf_config", list(
    python = normalizePath(python),
    libpython = normalizePath(libpython, mustWork = FALSE),
    pythonhome = pythonhome,
    version_string = version_string,
    version = version,
    anaconda = anaconda,
    numpy = numpy,
    tensorflow = tensorflow,
    python_versions = normalizePath(python_versions)
  ))

}

#' @export
print.tf_config <- function(x, ...) {
  cat("python:        ", x$python, "\n")
  cat("libpython:     ", x$libpython, ifelse(file.exists(x$libpython), "", "[NOT FOUND]"), "\n")
  cat("pythonhome:    ", x$pythonhome, "\n")
  cat("version:       ", x$version_string, "\n")
  if (!is.null(x$numpy)) {
    cat("numpy:         ", x$numpy$path, "\n")
    cat("numpy_version: ", as.character(x$numpy$version), "\n")
  } else {
    cat("numpy:          [NOT FOUND]\n")
  }
  if (!is.null(x$tensorflow)) {
    cat("tf:            ", x$tensorflow$path, "\n")
    cat("tf_version:    ", as.character(x$tensorflow$version), "\n")
  } else {
    cat("tf:             [NOT FOUND]\n")
  }
  if (length(x$python_versions) > 0) {
    cat("\npython versions found: \n")
    python_versions <- paste0(" ", x$python_versions)
    cat(python_versions, sep = "\n")
  }
}


is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}

is_osx <- function() {
  Sys.info()["sysname"] == "Darwin"
}

clean_tf_version <- function(tf_version) {
  gsub("\\.$", "", gsub("[A-Za-z_]+", "", tf_version))
}

tensorflow_python <- function() {

  # determine the location of python
  tensorflow_python <- Sys.getenv("TENSORFLOW_PYTHON", unset = NA)
  if (!is.na(tensorflow_python)) {

    # normalize trailing slash and expand
    tensorflow_python <- gsub("[\\/]+$", "", tensorflow_python)
    tensorflow_python <- path.expand(tensorflow_python)

    # check for existence
    if (!file_test("-d", tensorflow_python) && !file_test("-f", tensorflow_python)) {
      list(
        python = tensorflow_python,
        exists = FALSE
      )
    } else {

      # append binary if it's a directory
      if (file_test("-d", tensorflow_python))
        tensorflow_python <- file.path(tensorflow_python, "python")

      # append .exe if necessary on windows
      if (is_windows() && (!endsWith(tolower(tensorflow_python), ".exe")))
        tensorflow_python <- paste0(tensorflow_python, ".exe")

      # return
      list(
        python = tensorflow_python,
        exists = TRUE
      )
    }


  } else {
    NULL
  }
}

windows_registry_python_versions <- function() {

  read_python_versions <- function(hive) {
    versions <- c()
    python_core_key <- tryCatch(utils::readRegistry(
      key = "SOFTWARE\\Python\\PythonCore", hive = hive, maxdepth = 3),
      error = function(e) NULL)

    if (length(python_core_key) > 0) {
      for (version in names(python_core_key)) {
        version_key <- python_core_key[[version]]
        if (!is.null(version_key$InstallPath)) {
          version_dir <- version_key$InstallPath$`(Default)`
          version_dir <- gsub("[\\/]+$", "", version_dir)
          version_exe <- paste0(version_dir, "\\python.exe")
          versions <- c(versions, version_exe)
        }
      }
    }

    versions
  }

  c(read_python_versions("HCU"), read_python_versions("HLM"))
}


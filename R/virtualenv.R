


#' Interface to virtualenv
#' 
#' R functions for managing Python [virtual environments](https://virtualenv.pypa.io/en/stable/)
#' 
#' @details 
#' Virtual environments are by default located at `~/.virtualenvs`. You can change this 
#' behavior by defining the `WORKON_HOME` environment variable.
#' 
#' Virtual environment functions are not supported on Windows (the use of 
#' [conda environments][conda-tools] is recommended on Windows).
#' 
#' @param envname Name of virtual environment
#' @param packages Character vector with package names to install or remove.
#' @param ignore_installed Ignore any previously installed versions of packages  
#' @param confirm Confirm before removing packages or virtual environments 
#' 
#' @return `virtualenv_list()` returns a chracter vector with the names 
#'  of available virtual environments. `virtualenv_root()` returns the 
#'  root directory for virtual environments.
#' 
#' 
#' @name virtualenv-tools
#' @export
virtualenv_list <- function() {
  
  config <- virtualenv_config()
  
  if (utils::file_test("-d", config$root)) {
    list.files(config$root) 
  } else {
    character()
  }
}

#' @rdname virtualenv-tools
#' @export
virtualenv_root <- function() {
  Sys.getenv("WORKON_HOME", unset = "~/.virtualenvs")
}

#' @rdname virtualenv-tools
#' @export
virtualenv_create <- function(envname) {
  
  config <- virtualenv_config()
  
  virtualenv_path <- file.path(config$root, envname)
  virtualenv_bin <- function(bin) path.expand(file.path(virtualenv_path, "bin", bin))
  
  if (!utils::file_test("-d", virtualenv_path) || !file.exists(virtualenv_bin("activate"))) {
    cat("Creating virtualenv at ", virtualenv_path, "\n")
    result <- system2(config$virtualenv, shQuote(c(
      "--system-site-packages",
      "--python", config$python,
      path.expand(virtualenv_path)))
    )
    if (result != 0L)
      stop("Error ", result, " occurred creating virtualenv at ", virtualenv_path,
           call. = FALSE)
  } else {
    cat("virtualenv:", virtualenv_path, "\n")
  }

  invisible(NULL)
}

#' @rdname virtualenv-tools
#' @export
virtualenv_install <- function(envname, packages, ignore_installed = FALSE) {
  
  virtualenv_create(envname)
  
  config <- virtualenv_config()
  virtualenv_path <- file.path(config$root, envname)
  virtualenv_bin <- function(bin) path.expand(file.path(virtualenv_path, "bin", bin))
  
  # function to call pip within virtual env
  pip_install <- function(packages, message, ignore_installed_package) {
    cmd <- sprintf("%ssource %s && %s install %s --upgrade %s%s",
                   ifelse(is_osx(), "", "/bin/bash -c \""),
                   shQuote(path.expand(virtualenv_bin("activate"))),
                   shQuote(path.expand(virtualenv_bin(config$pip_version))),
                   ifelse(ignore_installed_package, "--ignore-installed", ""),
                   paste(shQuote(packages), collapse = " "),
                   ifelse(is_osx(), "", "\""))
    cat(message, "...\n")
    result <- system(cmd)
    if (result != 0L)
      stop("Error ", result, " occurred installing packages", call. = FALSE)
    invisible(NULL)
  }
  
  # upgrade pip so it can find tensorflow
  pip_install("pip", "Upgrading pip", TRUE)
  
  # install updated version of the wheel package
  pip_install("wheel", "Upgrading wheel", TRUE)
  
  # upgrade setuptools so it can use wheels
  pip_install("setuptools", "Upgrading setuptools", TRUE)
  
  # install packages
  pip_install(packages, "Installing packages", ignore_installed)
}


#' @rdname virtualenv-tools
#' @export
virtualenv_remove <- function(envname, packages = NULL, confirm = interactive()) {
  
  config <- virtualenv_config()
  virtualenv_path <- file.path(config$root, envname)
  virtualenv_bin <- function(bin) path.expand(file.path(virtualenv_path, "bin", bin))
  
  # packages = NULL means remove the entire virtualenv
  if (is.null(packages)) {
    
    if (confirm) {
      prompt <- readline(sprintf("Remove virtualenv at %s? [Y/n]: ", virtualenv_path))
      if (nzchar(prompt) && tolower(prompt) != 'y')
        return(invisible(NULL))
    }
    
    unlink(virtualenv_path, recursive = TRUE)
    
  } else {
    
    # function to call pip within virtual env
    pip_uninstall <- function(packages) {
      cmd <- sprintf("%ssource %s && %s uninstall --yes %s%s",
                     ifelse(is_osx(), "", "/bin/bash -c \""),
                     shQuote(path.expand(virtualenv_bin("activate"))),
                     shQuote(path.expand(virtualenv_bin(config$pip_version))),
                     paste(shQuote(packages), collapse = " "),
                     ifelse(is_osx(), "", "\""))
      cat(sprintf("Uninstalling %s", paste(packages, sep = ", ")), "...\n")
      result <- system(cmd)
      if (result != 0L)
        stop("Error ", result, " occurred removing packages", call. = FALSE)
    }
    
    if (confirm) {
      prompt <- readline(sprintf("Remove %s from virtualenv %s? [Y/n]: ", 
                                 paste(packages, sep = ", "), virtualenv_path))
      if (nzchar(prompt) && tolower(prompt) != 'y')
        return(invisible(NULL))
    }
    
    pip_uninstall(packages)
    
  }
  
  invisible(NULL)
}


virtualenv_config <- function() {
  
  # not supported on windows
  if (is_windows()) {
    stop("virtualenv functions are not supported on windows (try conda environments instead)",
         call. = FALSE)
  }
  
  # find system python binary
  python <- python_unix_binary("python")
  if (is.null(python))
    stop("Unable to locate Python on this system.", call. = FALSE)
  
  # find required binaries
  pip <- python_unix_binary("pip")
  have_pip <- !is.null(pip)
  virtualenv <- python_unix_binary("virtualenv")
  have_virtualenv <- !is.null(virtualenv)
  
  # validate that we have the required tools for the method
  install_commands <- NULL
  if (is_osx()) {
    if (!have_pip)
      install_commands <- c(install_commands, "$ sudo /usr/bin/easy_install pip")
    if (!have_virtualenv) {
      if (is.null(pip))
        pip <- "/usr/local/bin/pip"
      install_commands <- c(install_commands, sprintf("$ sudo %s install --upgrade virtualenv", pip))
    }
    if (!is.null(install_commands))
      install_commands <- paste(install_commands, collapse = "\n")
  } else if (is_ubuntu()) {
    if (!have_pip)
      install_commands <- c(install_commands, "python-pip")
    if (!have_virtualenv)
      install_commands <- c(install_commands, "python-virtualenv")
    if (!is.null(install_commands)) {
      install_commands <- paste("$ sudo apt-get install",
                                paste(install_commands, collapse = " "))
    }
  } else {
    if (!have_pip)
      install_commands <- c(install_commands, "pip")
    if (!have_virtualenv)
      install_commands <- c(install_commands, "virtualenv")
    if (!is.null(install_commands)) {
      install_commands <- paste("Please install the following Python packages before proceeding:",
                                paste(install_commands, collapse = ", "))
    }
  }
  if (!is.null(install_commands)) {
    
    # if these are terminal commands then add special preface
    if (grepl("^\\$ ", install_commands)) {
      install_commands <- paste0(
        "Execute the following at a terminal to install the prerequisites:\n\n",
        install_commands
      )
    }
    
    stop("Prerequisites for using Python virtualenvs not available.\n\n",
         install_commands, "\n\n", call. = FALSE)
  }
  
  # return config
  list(
    python = python,
    virtualenv = virtualenv,
    pip_version = ifelse(python_version(python) >= "3.0", "pip3", "pip"),
    root = virtualenv_root()
  )
}

python_unix_binary <- function(bin) {
  locations <- file.path(c("/usr/bin", "/usr/local/bin", path.expand("~/.local/bin")), bin)
  locations <- locations[file.exists(locations)]
  if (length(locations) > 0)
    locations[[1]]
  else
    NULL
}

python_version <- function(python) {
  
  # check for the version
  result <- system2(python, "--version", stdout = TRUE, stderr = TRUE)
  
  # check for error
  error_status <- attr(result, "status")
  if (!is.null(error_status))
    stop("Error ", error_status, " occurred while checking for python version", call. = FALSE)
  
  # parse out the major and minor version numbers
  matches <- regexec("^[^ ]+\\s+(\\d+)\\.(\\d+).*$", result)
  matches <- regmatches(result, matches)[[1]]
  if (length(matches) != 3)
    stop("Unable to parse Python version '", result[[1]], "'", call. = FALSE)
  
  # return as R numeric version
  numeric_version(paste(matches[[2]], matches[[3]], sep = "."))
}



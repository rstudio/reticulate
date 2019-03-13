#' Interface to conda
#'
#' R functions for managing Python [conda
#' environments](https://conda.io/docs/user-guide/tasks/manage-environments.html).
#'
#' @param envname Name of conda environment
#' @param conda Path to conda executable (or "auto" to find conda using the
#'   PATH and other conventional install locations).
#' @param packages Character vector with package names to install or remove.
#' @param pip `TRUE` to use pip (defaults to `FALSE`)
#'
#' @return `conda_list()` returns a data frame with the names and paths to the
#'   respective python binaries of available environments. `conda_create()`
#'   returns the Path to the python binary of the created environment.
#'   `conda_binary()` returns the location of the main conda binary or `NULL`
#'   if none can be found.
#'
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
conda_create <- function(envname = NULL, packages = "python", conda = "auto") {

  # resolve conda binary
  conda <- conda_binary(conda)

  # resolve environment name
  envname <- condaenv_resolve(envname)

  # create the environment
  args <- conda_args("create", envname, packages)
  result <- system2(conda, shQuote(args))
  if (result != 0L) {
    stop("Error ", result, " occurred creating conda environment ", envname,
         call. = FALSE)
  }

  # return the path to the python binary
  conda_python(envname = envname, conda = conda)

}

#' @rdname conda-tools
#' @export
conda_remove <- function(envname, packages = NULL, conda = "auto") {

  # resolve conda binary
  conda <- conda_binary(conda)

  # resolve environment name
  envname <- condaenv_resolve(envname)

  # no packages means everything
  if (is.null(packages))
    packages <- "--all"

  # remove packges (or the entire environment)
  args <- conda_args("remove", envname, packages)
  result <- system2(conda, shQuote(args))
  if (result != 0L) {
    stop("Error ", result, " occurred removing conda environment ", envname,
         call. = FALSE)
  }
}

#' @param forge Include the [Conda Forge](https://conda-forge.org/) repository.
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
conda_install <- function(envname = NULL, packages, forge = TRUE, pip = FALSE, pip_ignore_installed = TRUE, conda = "auto") {

  # resolve conda binary
  conda <- conda_binary(conda)

  # resolve environment name
  envname <- condaenv_resolve(envname)

  # create the environment if needed
  python <- conda_python(envname = envname, conda = conda)
  if (!file.exists(python))
    conda_create(envname, conda = conda)

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
    # use conda
    args <- conda_args("install", envname)
    if (forge)
      args <- c(args, "-c", "conda-forge")
    args <- c(args, packages)
    result <- system2(conda, shQuote(args))
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

  # if the user has requested a conda binary in the 'condabin' folder,
  # try to find and use its sibling in the 'bin' folder instead as
  # we rely on other tools typically bundled in the 'bin' folder
  # https://github.com/rstudio/keras/issues/691
  if (!is_windows()) {
    altpath <- file.path(dirname(conda), "../bin/conda")
    if (file.exists(altpath))
      return(normalizePath(altpath, winslash = "/", mustWork = TRUE))
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

#' @rdname conda-tools
#' @export
conda_python <- function(envname = NULL, conda = "auto") {

  # resolve envname
  envname <- condaenv_resolve(envname)

  # for fully-qualified paths, construct path explicitly
  if (grepl("[/\\]", envname, fixed = TRUE)) {
    suffix <- if (is_windows()) "python.exe" else "bin/python"
    path <- file.path(envname, suffix)
    if (file.exists(path))
      return(path)

    fmt <- "no conda environment exists at path '%s'"
    stop(sprintf(fmt, envname))
  }

  # otherwise, list conda environments and try to find it
  conda_envs <- conda_list(conda = conda)
  env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(env) > 0)
    path.expand(env$python[[1]])
  else
    stop("conda environment ", envname, " not found")
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

condaenv_resolve <- function(envname = NULL) {

  # handle case where envname is NULL (use default / active env)
  if (is.null(envname)) {

    default <- Sys.getenv("RETICULATE_PYTHON_ENV", unset = NA)
    if (!is.na(default)) {
      path <- normalizePath(default, winslash = "/", mustWork = FALSE)
      if (!is_condaenv(path)) {
        fmt <- "there is no conda environment at path '%s'"
        stop(sprintf(fmt, path))
      }
      return(path)
    }

    # provide context of caller (if any) when emitting error
    call <- sys.call(sys.parent())
    if (is.null(call))
      call <- sys.call()

    fmt <- "missing environment in call to '%s'"
    stop(sprintf(fmt, format(sys.call(sys.parent()))), call. = FALSE)

  }

  # treat environment 'names' containing slashes as paths
  # rather than environments living in WORKON_HOME
  if (grepl("[/\\]", envname)) {
    if (file.exists(envname))
      envname <- normalizePath(envname, winslash = "/")
    return(envname)
  }

  # no slashes; just use the environment name as-is
  envname

}

conda_args <- function(action, envname = NULL, ...) {

  envname <- condaenv_resolve(envname)

  # use '--prefix' as opposed to '--name' if envname looks like a path
  args <- c(action, "--yes")
  if (grepl("[/\\]", envname))
    args <- c(args, "--prefix", envname, ...)
  else
    args <- c(args, "--name", envname, ...)

  args

}

is_condaenv <- function(dir) {
  file.exists(file.path(dir, "conda-meta"))
}

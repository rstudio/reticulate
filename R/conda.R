

#' @param envname The name of, or path to, a conda environment.
#' 
#' @param conda The path to a `conda` executable. Use `"auto"` to allow
#'   `reticulate` to automatically find an appropriate `conda` binary. See
#'   [conda_binary()] for more details on how `reticulate` tries to resolve
#'   the `conda` executable.
#'
#' @param forge Boolean; include the [conda-forge](https://conda-forge.org/)
#'   repository?
#'   
#' @param channel An optional character vector of conda channels to include.
#'   When specified, the `forge` argument is ignored. If you need to
#'   specify multiple channels, including the conda Forge, you can use
#'   `c("conda-forge", <other channels>)`.
#'
#' @name conda-params
NULL

#' List Conda Environments
#' 
#' List all of the available conda environments on the system.
#' 
#' Environments available are listed as by:
#' 
#' ```
#' conda info --json
#' ```
#' 
#' 
#' Note that this function will report _all_ available conda environments
#' on the system, even those associated with a different conda installation
#' than the requested `conda` binary.
#' 
#' @inheritParams conda-params
#' 
#' @return An \R `data.frame`, with `name` giving the name of the associated
#'   environment, and `python` giving the path to the Python binary associated
#'   with that environment.
#'
#' @family conda tools
#' @export
conda_list <- function(conda = "auto") {

  # resolve conda binary
  conda <- conda_binary(conda)

  # list envs -- discard stderr as Anaconda may emit warnings that can
  # otherwise be ignored; see e.g. https://github.com/rstudio/reticulate/issues/474
  conda_envs <- suppressWarnings(
    system2(conda, args = c("info", "--json"), stdout = TRUE, stderr = FALSE)
  )

  # check for error
  status <- attr(conda_envs, "status") %||% 0L
  if (status != 0L) {
  
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
  
  # parse conda info
  info <- fromJSON(conda_envs)

  # convert to json
  conda_envs <- info$envs
  
  # normalize and remove duplicates (seems necessary on Windows as Anaconda
  # may report both short-path and long-path versions of the same environment)
  conda_envs <- unique(normalizePath(conda_envs, mustWork = FALSE))
  
  # return an empty data.frame when no envs are found
  if (length(conda_envs) == 0L) {
    return(data.frame(
      name = character(),
      python = character(),
      stringsAsFactors = FALSE)
    )
  }

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
  
  data.frame(
    name = name,
    python = python,
    stringsAsFactors = FALSE
  )
  
}


#' Create a Conda Environment
#' 
#' Create a new conda environment.
#' 
#' @inheritParams conda-params
#' 
#' @param environment The path to an environment definition, generated via
#'   (for example) [conda_export()], or via `conda env export`. When provided,
#'   the conda environment will be created using this environment definition,
#'   and other arguments will be ignored.
#'   
#' @param python_version The version of Python to be used in this conda
#'   environment. The associated Python package from conda will be requested
#'   as `python={python_version}`. When `NULL`, the default `python` package
#'   will be used instead. For example, use `python_version = "3.6"` to request
#'   that the conda environment be created with a copy of Python 3.6. This
#'   argument will be ignored if `python` is specified as part of the `packages`
#'   argument, for backwards compatibility.
#'   
#' @return The path to the Python binary associated with the newly-created
#'   conda environment.
#'
#' @family conda tools
#' @export
conda_create <- function(envname = NULL,
                         packages = NULL,
                         ...,
                         forge = TRUE,
                         channel = character(),
                         environment = NULL,
                         conda = "auto",
                         python_version = NULL)
{
  # resolve conda binary
  conda <- conda_binary(conda)

  # if environment is provided, use it directly
  if (!is.null(environment))
    return(conda_create_env(envname, environment, conda))
  
  # resolve environment name
  envname <- condaenv_resolve(envname)
  
  # resolve packages argument
  if (!any(grepl("^python", packages))) {
    
    python_package <- if (is.null(python_version))
      "python"
    else
      sprintf("python=%s", python_version)
    
    packages <- c(python_package, packages)
    
  }

  # create the environment
  args <- conda_args("create", envname, packages)

  # be quiet
  args <- c(args, "--quiet")
  
  # add user-requested channels
  channels <- if (length(channel))
    channel
  else if (forge)
    "conda-forge"

  for (ch in channels)
    args <- c(args, "-c", ch)
  
  # invoke conda
  result <- system2(conda, shQuote(args))
  if (result != 0L) {
    fmt <- "Error creating conda environment '%s' [exit code %i]"
    stopf(fmt, envname, result, call. = FALSE)
  }

  # return the path to the python binary
  conda_python(envname = envname, conda = conda)
}

conda_create_env <- function(envname, environment, conda) {
 
  if (!is.null(envname))
    envname <- condaenv_resolve(envname)
  
  args <- c(
    "env", "create", "--quiet",
    if (is.null(envname))
      c()
    else if (grepl("/", envname))
      c("--prefix", shQuote(envname))
    else
      c("--name", shQuote(envname)),
    "-f", shQuote(environment)
  )
  
  result <- system2(conda, args)
  if (result != 0L) {
    fmt <- "Error creating conda environment [exit code %i]"
    stopf(fmt, result)
  }
  
  # return the path to the python binary
  conda_python(envname = envname, conda = conda)
  
}

#' Export a Conda Environment
#' 
#' Export a conda environment definition, either as YAML (the default)
#' or JSON. The resulting environment file can be used by [conda_create()]
#' to create a "clone" of the exported conda environment.
#' 
#' @inheritParams conda-params
#' 
#' @param file The path where the conda environment definition will be written.
#' 
#' @param json Boolean; should the environment definition be written as JSON?
#'   By default, conda exports environmentas as YAML.
#'
#' @param ... Optional arguments, currently ignored.
#' 
#' @return The path to the exported environment definition, invisibly.
#' 
#' @family conda tools
#' @export
conda_export <- function(envname,
                         file = if (json) "environment.json" else "environment.yml",
                         json = FALSE,
                         ...,
                         conda = "auto")
{
  # resolve parameters
  conda <- conda_binary(conda)
  envname <- condaenv_resolve(envname)
  
  # build conda argument list,
  args <- c(
    "env", "export",
    if (json)
      "--json",
    if (grepl("/", envname))
      c("--prefix", shQuote(envname))
    else
      c("--name", shQuote(envname)),
    ">", shQuote(file)
  )
  
  # execute conda
  status <- system2(conda, args)
  if (status != 0L) {
    fmt <- "Error exporting conda environment [error code %i]"
    stopf(fmt, status, call. = FALSE)
  }
  
  # notify user
  fmt <- "* Environment '%s' exported to '%s'."
  writeLines(sprintf(fmt, envname, file))
  
  # return path to generated environment
  invisible(file)
}

#' Remove Packages from a Conda Environment
#' 
#' Use this function to remove some subset of packages from an existing
#' conda environment, or (when `packages` is `NULL`) remove all packages
#' from the requested conda environment.
#' 
#' @inheritParams conda-params
#' 
#' @param packages An optional character vector, giving the names of packages
#'   to be removed from the conda environment. When `NULL` (the default),
#'   everything in the associated environment will be removed.
#'
#' @family conda tools
#' @export
conda_remove <- function(envname,
                         packages = NULL,
                         conda = "auto")
{
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

#' Install a Package in a Conda Environment
#' 
#' Install a package into a conda environment. Packages can be installed either
#' from the conda repositories, or via `pip`.
#' 
#' @inheritParams conda-params
#' 
#' @param pip Boolean; use `pip` for package installation? By default, packages
#'   are installed from the active conda channels.
#'
#' @param pip_ignore_installed Ignore already-installed versions when using pip?
#'   (defaults to `FALSE`). Set this to `TRUE` so that specific package versions
#'   can be installed even if they are downgrades. The `FALSE` option is useful
#'   for situations where you don't want a pip install to attempt an overwrite of
#'   a conda binary package (e.g. SciPy on Windows which is very difficult to
#'   install via pip due to compilation requirements).
#'   
#' @param pip_options An optional character vector of additional command line
#'   arguments to be passed to `pip`. Only relevant when `pip = TRUE`.
#'   
#' @param python_version The version of Python to be installed. Set this if
#'   you'd like to change the version of Python associated with a particular
#'   conda environment.
#'
#' @family conda tools
#' @export
conda_install <- function(envname = NULL,
                          packages,
                          forge = TRUE,
                          channel = character(),
                          pip = FALSE,
                          pip_options = character(),
                          pip_ignore_installed = FALSE,
                          conda = "auto",
                          python_version = NULL,
                          ...)
{
  check_forbidden_install("Python packages")
  
  # check that 'packages' argument was supplied
  if (missing(packages)) {
    if (!is.null(envname)) {
      
      fmt <- paste(
        "argument \"packages\" is missing, with no default",
        "- did you mean 'conda_install(<envname>, %1$s)'?",
        "- use 'py_install(%1$s)' to install into the active Python environment",
        sep = "\n"
      )
      
      stopf(fmt, deparse1(substitute(envname)), call. = FALSE)
    } else {
      packages
    }
  }
  
  # resolve conda binary
  conda <- conda_binary(conda)

  # resolve environment name
  envname <- condaenv_resolve(envname)

  # honor request for specific version of Python package
  python_package <- if (is.null(python_version))
    NULL
  else if (grepl("[><=]", python_version))
    paste0("python", python_version)
  else
    sprintf("python=%s", python_version)

  # check to see if we already have a valid Python installation for
  # this conda environment
  python <- tryCatch(
    conda_python(envname = envname, conda = conda),
    error = identity
  )
  
  # if this conda environment doesn't seem to exist, auto-create it
  if (inherits(python, "error") || !file.exists(python)) {
    
    conda_create(
      envname = envname,
      packages = python_package %||% "python",
      forge = forge,
      channel = channel,
      conda = conda
    )
    
    python <- conda_python(envname = envname, conda = conda)
    
  }
  
  # if the user has requested a specific version of Python, ensure that
  # version of Python is installed into the requested environment
  # (should be no-op if that copy of Python already installed)
  if (!is.null(python_version)) {
    args <- conda_args("install", envname, python_package)
    status <- system2(conda, shQuote(args))
    if (status != 0L) {
      fmt <- "installation of '%s' into environment '%s' failed [error code %i]"
      msg <- sprintf(fmt, python_package, envname, status)
      stop(msg, call. = FALSE)
    }
  }

  # delegate to pip if requested
  if (pip) {
    
    result <- pip_install(
      python = python,
      packages = packages,
      pip_options = pip_options,
      ignore_installed = pip_ignore_installed,
      conda = conda,
      envname = envname
    )
    
    return(result)
    
  }
  
  # otherwise, use conda
  args <- conda_args("install", envname)
  
  # add user-requested channels
  channels <- if (length(channel))
    channel
  else if (forge)
    "conda-forge"
  
  for (ch in channels)
    args <- c(args, "-c", ch)
    
  args <- c(args, python_package, packages)
  result <- system2(conda, shQuote(args))
  
  # check for errors
  if (result != 0L) {
    fmt <- "one or more Python packages failed to install [error code %i]"
    stopf(fmt, result)
  }

  
  invisible(packages)
}

#' Find a Conda Executable
#' 
#' Locate a conda installation on the machine, and find the associated path to
#' the `conda` executable within that installation.
#' 
#' @section Finding Conda:
#' 
#' Most of `reticulate`'s conda APIs accept a `conda` parameter, used to control
#' the `conda` binary used in their operation. When `conda = "auto"`,
#' `reticulate` will attempt to automatically find a conda installation.
#' The following locations are searched, in order:
#' 
#' 1. The location specified by the `reticulate.conda_binary` \R option,
#' 2. The location specified by the `RETICULATE_CONDA` environment variable,
#' 3. The [miniconda_path()] location (if it exists),
#' 4. The program `PATH`,
#' 5. A set of pre-defined locations where conda is typically installed.
#' 
#' To force `reticulate` to use a particular `conda` binary, we recommend
#' setting:
#' 
#' ```
#' options(reticulate.conda_binary = "/path/to/conda")
#' ```
#' 
#' This can be useful if your conda installation lives in a location that
#' `reticulate` is unable to automatically discover.
#' 
#' @inheritParams conda-params
#' 
#' @family conda tools
#' @export
conda_binary <- function(conda = "auto") {

  # automatic lookup if requested
  if (identical(conda, "auto")) {
    conda <- find_conda()
    if (is.null(conda))
      stop("Unable to find conda binary. Is Anaconda installed?", call. = FALSE)
    conda <- conda[[1]]
  }

  conda <- normalizePath(conda, winslash = "/", mustWork = FALSE)
  
  if(!grepl("^conda", basename(conda)))
    stop("Supplied path is not a conda binary: ", sQuote(conda, FALSE))
  
  # if the user has requested a conda binary in the 'condabin' folder,
  # try to find and use its sibling in the 'bin' folder instead as
  # we rely on other tools typically bundled in the 'bin' folder
  # https://github.com/rstudio/keras/issues/691
  if (!is_windows()) {
    altpath <- file.path(dirname(conda), "../bin/conda")
    if (file.exists(altpath))
      return(normalizePath(altpath, winslash = "/", mustWork = TRUE))
  } else {
    # on Windows it's preferable to conda.bat located in the 'condabin'
    # folder. if the user passed the path to a 'Scripts/conda.exe' we will
    # try to find the 'conda.bat'.
    altpath <- file.path(dirname(conda), "../condabin/conda.bat")
    if (file.exists(altpath))
      return(normalizePath(altpath, winslash = "/", mustWork = TRUE))
  }

  # validate existence
  if (!file.exists(conda))
    stop("Specified conda binary '", conda, "' does not exist.", call. = FALSE)

  # return conda
  conda
}

#' @rdname conda_binary
#' @export
conda_exe <- conda_binary


#' Retrieve the Conda Version
#' 
#' Retrieve the current version of conda, as reported by `conda --version`.
#' 
#' @inheritParams conda-params
#' 
#' @family conda tools
#' @export
conda_version <- function(conda = "auto") {
  conda_bin <- conda_binary(conda)
  system2(conda_bin, "--version", stdout = TRUE)
}


numeric_conda_version <- function(conda = "auto", version_string = conda_version(conda)) {
  # some plausible version strings: 
  # "conda 4.6.0"                 
  # "conda 4.6.0b0"               
  # "conda 4.6.0rc1"              
  # "conda 4.6.0rc1.post3+64bde06"
  v <- version_string 
  v <- sub("^conda ", "", v) # drop hardcoded prefix
  
  # https://github.com/conda/conda/blob/c1579681d1468af3d1b4af3083bed33f8391e861/conda/_vendor/auxlib/packaging.py#L142
  # if dev version string: "{0}.post{1}+{2}".format(version, post_commit, hash)
  v <- sub("\\.post(\\d)\\+.+$", ".\\1", v)

  # substitute rc|beta|alpha|whatever suffix with .
  v <- sub("[A-Za-z]+", ".", v) 
  
  if (grepl("\\.$", v))
    v <- paste0(v, "9000")
  
  numeric_version(v)
}


#' Get Python Path in Conda Environment
#' 
#' Find the path to the `python` executable associated with a particular
#' conda environment.
#' 
#' @inheritParams conda-params
#' 
#' @family conda tools
#' @export
conda_python <- function(envname = NULL, conda = "auto") {

  # resolve envname
  envname <- condaenv_resolve(envname)

  # for fully-qualified paths, construct path explicitly
  if (grepl("[/\\\\]", envname)) {
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
  
  # allow specification of conda executable
  conda <- getOption("reticulate.conda_binary")
  if (!is.null(conda))
    return(conda)
  
  conda <- Sys.getenv("RETICULATE_CONDA", unset = NA)
  if (!is.na(conda))
    return(conda)
  
  # if miniconda is installed, use it
  if (miniconda_exists())
    return(miniconda_conda())
  
  # if there is a conda executable on the PATH, use it
  conda <- Sys.which("conda")
  if (nzchar(conda))
    return(conda)
  
  # otherwise, search common locations for conda
  prefixes <- c("~/opt/", "~/", "/opt/", "/")
  names <- c("anaconda", "miniconda", "miniforge")
  versions <- c("", "2", "3", "4")
  combos <- expand.grid(versions, names, prefixes, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  combos <- combos[rev(seq_along(combos))]
  conda_locations <- unlist(.mapply(paste0, combos, NULL))
  
  # find the potential conda binary path in each case
  conda_locations <- if (is_windows()) {
    paste0(conda_locations, "/condabin/conda.bat")
  } else {
    paste0(conda_locations, "/bin/conda")
  }
  
  # ensure we expand tilde prefixes
  conda_locations <- path.expand(conda_locations)
    
  # on Windows, check the registry for a compatible version of Anaconda
  if (is_windows()) {
    anaconda_versions <- windows_registry_anaconda_versions()
    anaconda_versions <- subset(anaconda_versions, anaconda_versions$arch == .Platform$r_arch)
    if (nrow(anaconda_versions) > 0) {
      
      conda_scripts <- utils::shortPathName(
        file.path(anaconda_versions$install_path, "Scripts", "conda.exe")
      )
      conda_bats <- utils::shortPathName(
        file.path(anaconda_versions$install_path, "condabin", "conda.bat")
      )
      
      # prefer versions found in the registry to those found in default locations
      conda_locations <- c(conda_bats, conda_scripts, conda_locations)
      
    }
  }
  
  # keep only conda locations that exist
  conda_locations <- conda_locations[file.exists(conda_locations)]
  if (length(conda_locations))
    return(conda_locations)
  
  # explicitly return NULL when no conda found
  NULL
  
}

condaenv_resolve <- function(envname = NULL) {

  python_environment_resolve(
    envname = envname,
    resolve = identity
  )

}

condaenv_exists <- function(envname = NULL, conda = "auto") {

  # check that conda is installed
  condabin <- tryCatch(conda_binary(conda = conda), error = identity)
  if (inherits(condabin, "error"))
    return(FALSE)

  # check that the environment exists
  python <- tryCatch(conda_python(envname, conda = conda), error = identity)
  if (inherits(python, "error"))
    return(FALSE)

  # validate the Python binary exists
  file.exists(python)

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

conda_list_packages <- function(envname = NULL, conda = "auto", no_pip = TRUE) {
  
  conda <- conda_binary(conda)
  envname <- condaenv_resolve(envname)

  # create the environment
  args <- c("list")
  if (grepl("[/\\]", envname)) {
    args <- c(args, "--prefix", envname)
  } else {
    args <- c(args, "--name", envname)
  }
  
  if (no_pip)
    args <- c(args, "--no-pip")
  
  args <- c(args, "--json")
  
  output <- system2(conda, shQuote(args), stdout = TRUE)
  status <- attr(output, "status") %||% 0L
  if (status != 0L) {
    fmt <- "error listing conda environment [status code %i]"
    stopf(fmt, status)
  }
  
  parsed <- jsonlite::fromJSON(output)
  
  data.frame(
    package     = parsed$name,
    version     = parsed$version,
    requirement = paste(parsed$name, parsed$version, sep = "="),
    channel     = parsed$channel,
    stringsAsFactors = FALSE
  )
  
}

conda_installed <- function() {
  condabin <- tryCatch(conda_binary(), error = identity)
  if (inherits(condabin, "error"))
    return(FALSE)
  else
    return(TRUE)
}


conda_run <- function(cmd, args = c(), conda = "auto", envname = NULL,
                      run_args = c("--no-capture-output"), ...) {

  conda <- conda_binary(conda)
  envname <- condaenv_resolve(envname)

  if (numeric_conda_version(conda) < "4.9")
    stopf(
"`conda_run()` requires conda version >= 4.9.
Run `miniconda_update('%s')` to update conda.", conda)


  if(grepl("[/\\]", envname))
    in_env <- c("--prefix", shQuote(normalizePath(envname)))
  else
    in_env <- c("--name", envname)

  system2(conda, c("run", in_env, run_args,
                   shQuote(cmd), args), ...)
}

# executes a cmd with a conda env active, implemented directly to avoid using `conda run`
# https://github.com/conda/conda/issues/10972
conda_run2 <- function(...) {
  if(is_windows())
    conda_run2_windows(...)
  else
    conda_run2_nix(...)
}

conda_run2_windows <- function(cmd, args = c(), conda = "auto", envname = NULL) {
  conda <- normalizePath(conda_binary(conda))
  
  if(identical(envname, "base"))
    envname <- file.path(dirname(conda), "../..")
  else
    envname <- condaenv_resolve(envname)
  
  if(grepl("[/\\]", envname))
    envname <- normalizePath(envname)
  
  fi <- tempfile(fileext = ".bat")
  on.exit(unlink(fi))
  writeLines(c(
    paste("CALL", shQuote(conda), "activate", shQuote(envname)),
    paste(shQuote(cmd), paste(args, collapse = " "))
  ), fi)

  shell(fi)
}

conda_run2_nix <- function(cmd, args = c(), conda = "auto", envname = NULL) {
  conda <- normalizePath(conda_binary(conda))
  activate <- normalizePath(file.path(dirname(conda), "activate"))
  
  if(!identical(envname, "base")) {
    envname <- condaenv_resolve(envname)
    if (grepl("[/\\]", envname))
      envname <- normalizePath(envname)
  }
  
  fi <- tempfile(fileext = ".sh")
  on.exit(unlink(fi))
  writeLines(c(
    paste(".", activate),
    if(!identical(envname, "base"))
      paste("conda activate", shQuote(envname)),
    'echo "Activated conda python: $(which python)"',
    paste(shQuote(cmd), paste(args, collapse = " "))
  ), fi)
  system2(Sys.which("sh"), fi)
}


conda_info <- function(conda = "auto") {
  conda <- normalizePath(conda_binary(conda))
  json <- system2(conda, c("info", "--json"), stdout = TRUE)
  jsonlite::parse_json(json, simplifyVector = TRUE)
}

is_conda_python <- function(python) {
  root <- if (is_windows())
    dirname(python)
  else
    dirname(dirname(python))
  
  file.exists(file.path(root, "conda-meta"))
}


get_python_conda_info <- function(python) {
  stopifnot(is_conda_python(python))
  
  root <- if (is_windows()) 
    dirname(python)
  else
    dirname(dirname(python))
  
  if(dir.exists(file.path(root, "condabin"))) {
    # base conda env
    conda <- if(is_windows())
      file.path(root, "condabin/conda.bat")
    else
      file.path(root, "bin/conda")
  } else {
    # not base env, parse conda-meta history to find the conda binary
    # that created it
    conda <- python_info_condaenv_find(root)
  }
  
  list(conda = normalizePath(conda, winslash = "/", mustWork = TRUE),
       root = normalizePath(root, winslash = "/", mustWork = TRUE))
}
